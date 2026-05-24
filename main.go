package main

import (
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/nakagami/grdp"
	_ "github.com/nakagami/grdp/plugin/rdpgfx/ffmpeg"
	"github.com/nakagami/grdp/plugin/rdpsnd"
	"github.com/veandco/go-sdl2/sdl"
)

const sdlPixelFormatNV12 = uint32(0x3231564E) // SDL_PIXELFORMAT_NV12

// maxAudioQueueBytes is the soft cap on SDL2's queued audio buffer (≈1 s of
// PCM 44100 Hz / 2 ch / 16-bit).  When the queue exceeds this limit, incoming
// audio packets are dropped to prevent ever-growing latency.
const maxAudioQueueBytes = 176400

// h264DropCooldown is how long after the last dropped H.264 frame we keep
// signalling congestion to the server via the queueDepth hint.  Once this
// window elapses without a new drop, the hint is cleared and the server
// resumes full-quality encoding.
const h264DropCooldown = time.Second

// h264CongestionHint is the queueDepth value sent to the server while the
// SDL rendering pipeline is dropping H.264 frames.  The server interprets
// this as "20 frames are queued" and reduces H.264 bitrate/quality
// accordingly.  0 = no congestion, 0xFFFFFFFF = pause entirely.
const h264CongestionHint uint32 = 20

// maxBitmapBatchesPerTick limits how many bitmap batches are painted to the
// GPU texture in a single main-loop iteration.  Processing all pending batches
// at once (unlimited) can cause a stall spike when VirtualBox sends a large
// burst of updates.  Limiting to 1 spreads the GPU writes across render ticks,
// smoothing out latency at the cost of catching up slightly more slowly.
// Tune this value to taste: 1 = most even, higher = faster catch-up.
const maxBitmapBatchesPerTick = 1

// fillBytes sets every element of s to v using a doubling-copy strategy.
// copy() is implemented with SIMD instructions (NEON on ARM64, AVX on x86),
// so this is O(log n) SIMD copies instead of O(n) scalar writes —
// roughly 10–50× faster for the large UV chroma buffers (~1 MB at 1080p).
func fillBytes(s []byte, v byte) {
	if len(s) == 0 {
		return
	}
	s[0] = v
	for i := 1; i < len(s); i *= 2 {
		copy(s[i:], s[:i])
	}
}

// paintImages uploads each bitmap patch into the SDL2 streaming texture.
// Dirty rects are appended to dirtyRects so the caller can later clear only
// those regions (instead of the entire texture) when a new H.264 frame arrives.
func paintImages(bs []grdp.Bitmap, texture *sdl.Texture, dirtyRects *[]sdl.Rect) {
	for _, bm := range bs {
		// The texture uses PIXELFORMAT_BGRA32, so grdp's native BGRA data
		// (BitsPerPixel==4) can be passed directly — no copy or byte-swap needed.
		// For legacy bit-depths (2/3 bpp), bm.RGBA() converts to RGBA; we then
		// swap R↔B in-place to match the BGRA32 texture format.  Those paths
		// are uncommon outside of traditional RDP bitmap updates.
		if bm.BitsPerPixel == 4 {
			// Fast path: BGRA data passes straight to the BGRA32 texture.
			w := min(bm.DestRight-bm.DestLeft+1, bm.Width)
			h := min(bm.DestBottom-bm.DestTop+1, bm.Height)
			rect := sdl.Rect{X: int32(bm.DestLeft), Y: int32(bm.DestTop), W: int32(w), H: int32(h)}
			texture.Update(&rect, unsafe.Pointer(&bm.Data[0]), bm.Width*4)
			*dirtyRects = append(*dirtyRects, rect)
		} else {
			// Slow path: convert to RGBA, then swap R↔B for BGRA32 texture.
			// Use the smaller of the destination rectangle and the actual
			// image dimensions (same clamping as before).
			img := bm.RGBA()
			bounds := img.Bounds()
			w := min(bm.DestRight-bm.DestLeft+1, bounds.Dx())
			h := min(bm.DestBottom-bm.DestTop+1, bounds.Dy())
			p := img.Pix
			for i := 0; i < len(p); i += 4 {
				p[i], p[i+2] = p[i+2], p[i]
			}
			rect := sdl.Rect{X: int32(bm.DestLeft), Y: int32(bm.DestTop), W: int32(w), H: int32(h)}
			texture.Update(&rect, unsafe.Pointer(&img.Pix[0]), img.Stride)
			*dirtyRects = append(*dirtyRects, rect)
		}
	}
}

// uploadYUVFrame uploads a decoded H.264 YUV frame into the SDL2 YUV texture.
//
// On the SDL2 Metal renderer, SDL_UpdateNVTexture / SDL_UpdateYUVTexture each
// allocate a separate staging MTLTexture per plane and commit two independent
// Metal command buffers — a known inefficiency acknowledged by an SDL TODO
// comment (src/render/metal/SDL_render_metal.m:710-711).
//
// SDL_LockTexture on the Metal backend instead allocates a single lightweight
// MTLBuffer in shared (CPU+GPU unified) memory, lets us write both planes in
// one Go pass, and uploads everything in a single command-buffer commit on
// Unlock — halving GPU command overhead per frame.
//
// go-sdl2's Lock() computes the returned slice length as pitch×height (Y plane
// only), omitting the chroma plane.  We extend the slice with unsafe.Slice so
// we can write into the chroma region that SDL's MTLBuffer actually allocates.
//
// If Lock fails (e.g. software renderer fallback) we fall back to UpdateNV /
// UpdateYUV which are equivalent in correctness.
func uploadYUVFrame(frame yuvFrame, texture *sdl.Texture, rect *sdl.Rect) {
	ph := (frame.h + 1) / 2

	if frame.format == sdlPixelFormatNV12 {
		pixels, pitch, err := texture.Lock(rect)
		if err != nil {
			texture.UpdateNV(rect, frame.y, frame.yStride, frame.uv, frame.uvStride)
			return
		}
		defer texture.Unlock()
		yLen := pitch * frame.h
		uvLen := pitch * ph
		// Extend the Y-only slice to cover the full NV12 MTLBuffer (Y + interleaved UV).
		all := unsafe.Slice(&pixels[0], yLen+uvLen)
		if pitch == frame.yStride {
			copy(all[:yLen], frame.y[:yLen])
			copy(all[yLen:yLen+uvLen], frame.uv[:uvLen])
		} else {
			w := frame.w
			for row := 0; row < frame.h; row++ {
				copy(all[row*pitch:row*pitch+w], frame.y[row*frame.yStride:])
			}
			for row := range ph {
				copy(all[yLen+row*pitch:yLen+row*pitch+w], frame.uv[row*frame.uvStride:])
			}
		}
	} else {
		// I420 (IYUV): layout is Y | U | V with U/V each at half-width, half-height.
		pixels, pitch, err := texture.Lock(rect)
		if err != nil {
			texture.UpdateYUV(rect, frame.y, frame.yStride, frame.u, frame.uStride, frame.v, frame.vStride)
			return
		}
		defer texture.Unlock()
		yLen := pitch * frame.h
		uPitch := (pitch + 1) / 2
		uvLen := uPitch * ph
		// Extend slice to cover Y + U + V planes.
		all := unsafe.Slice(&pixels[0], yLen+uvLen+uvLen)
		if pitch == frame.yStride && uPitch == frame.uStride {
			copy(all[:yLen], frame.y[:yLen])
			copy(all[yLen:yLen+uvLen], frame.u[:uvLen])
			copy(all[yLen+uvLen:yLen+uvLen+uvLen], frame.v[:uvLen])
		} else {
			w := frame.w
			hw := (frame.w + 1) / 2
			for row := 0; row < frame.h; row++ {
				copy(all[row*pitch:row*pitch+w], frame.y[row*frame.yStride:])
			}
			for row := range ph {
				copy(all[yLen+row*uPitch:yLen+row*uPitch+hw], frame.u[row*frame.uStride:])
			}
			for row := range ph {
				copy(all[yLen+uvLen+row*uPitch:yLen+uvLen+row*uPitch+hw], frame.v[row*frame.vStride:])
			}
		}
	}
}

// audioPlayer manages SDL2 audio device for RDPSND playback.
// The device is opened once on the main thread at startup with a fixed output
// format (44100 Hz / stereo / S16LE).  play() converts incoming audio from
// whatever format the server negotiated (any PCM rate/channels/bit-depth) to
// the device format using SDL2's BuildAudioCVT/ConvertAudio, then enqueues the
// result.  All fields except deviceID are protected by mu and may be accessed
// from any goroutine.
//
// sdl.AudioCVT is intentionally NOT stored as a struct field: it embeds C
// function pointers (filter callbacks) that could interfere with Go's GC if
// kept on the heap between calls.  Instead it is allocated as a local variable
// inside play() for the duration of each conversion.
type audioPlayer struct {
	deviceID     sdl.AudioDeviceID
	reopenNeeded atomic.Bool  // set from play() on "Invalid audio device ID"; cleared by reopen() on main thread
	mu           sync.Mutex   // protects cvtKey, cvtNeed, cvtBuf
	cvtKey       [3]int       // [SamplesPerSec, Channels, BitsPerSample] of the last-probed CVT
	cvtNeed  bool       // true when the last-probed CVT actually transforms data
	cvtBuf   []byte     // reusable scratch buffer for in-place conversion
}

// open opens the audio device on the calling (main) thread.
func (a *audioPlayer) open() error {
	desired := sdl.AudioSpec{
		Freq:     44100,
		Format:   sdl.AUDIO_S16LSB,
		Channels: 2,
		Samples:  4096,
	}
	var obtained sdl.AudioSpec
	dev, err := sdl.OpenAudioDevice("", false, &desired, &obtained, 0)
	if err != nil {
		return err
	}
	a.deviceID = dev
	sdl.PauseAudioDevice(dev, false)
	slog.Debug("audio: opened device", "freq", obtained.Freq, "ch", obtained.Channels, "fmt", obtained.Format)
	return nil
}

// play converts audio from the server-negotiated format af to the device's
// fixed 44100 Hz / stereo / S16LE format (using SDL2 AudioCVT) and queues it
// for playback.  Incoming audio is dropped when the device queue is already
// near-full to prevent runaway latency.  Safe to call from any goroutine.
func (a *audioPlayer) play(af rdpsnd.AudioFormat, data []byte) {
	if a.deviceID == 0 {
		return
	}
	if sdl.GetQueuedAudioSize(a.deviceID) >= maxAudioQueueBytes {
		return // drop to prevent latency buildup
	}

	key := [3]int{int(af.SamplesPerSec), int(af.Channels), int(af.BitsPerSample)}
	// 8-bit PCM in WAVE is unsigned; 16-bit is signed little-endian.
	var srcFmt sdl.AudioFormat = sdl.AUDIO_S16LSB
	if af.BitsPerSample == 8 {
		srcFmt = sdl.AUDIO_U8
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Probe the CVT when the server changes format (rare, usually once per session).
	if key != a.cvtKey {
		a.cvtKey = key
		a.cvtNeed = false
		var probeCVT sdl.AudioCVT
		ok, err := sdl.BuildAudioCVT(&probeCVT, srcFmt, uint8(key[1]), key[0],
			sdl.AUDIO_S16LSB, 2, 44100)
		if err != nil {
			slog.Error("audio: BuildAudioCVT", "err", err, "src", af)
			return
		}
		a.cvtNeed = ok
		slog.Debug("audio: CVT", "src", af, "needs_cvt", ok)
	}

	if !a.cvtNeed {
		// Format already matches the device; queue directly.
		if err := sdl.QueueAudio(a.deviceID, data); err != nil {
			if strings.Contains(err.Error(), "Invalid audio device ID") {
				a.reopenNeeded.Store(true)
			} else {
				slog.Error("audio: QueueAudio", "err", err)
			}
		}
		return
	}

	// SDL_ConvertAudio works in-place: copy source into a buffer of size
	// len(data)*LenMult, set cvt.Buf/Len, call ConvertAudio, read LenCVT bytes.
	// Build a fresh local CVT (not stored on the heap) for each conversion so
	// the C filter-function pointers it contains are stack-lived only.
	var cvt sdl.AudioCVT
	if _, err := sdl.BuildAudioCVT(&cvt, srcFmt, uint8(key[1]), key[0],
		sdl.AUDIO_S16LSB, 2, 44100); err != nil {
		slog.Error("audio: BuildAudioCVT", "err", err, "src", af)
		return
	}
	needed := len(data) * int(cvt.LenMult)
	if cap(a.cvtBuf) < needed {
		a.cvtBuf = make([]byte, needed)
	}
	a.cvtBuf = a.cvtBuf[:needed]
	copy(a.cvtBuf, data)
	cvt.Buf = unsafe.Pointer(&a.cvtBuf[0])
	cvt.Len = int32(len(data))
	if err := sdl.ConvertAudio(&cvt); err != nil {
		slog.Error("audio: ConvertAudio", "err", err)
		return
	}
	if err := sdl.QueueAudio(a.deviceID, a.cvtBuf[:cvt.LenCVT]); err != nil {
		if strings.Contains(err.Error(), "Invalid audio device ID") {
			a.reopenNeeded.Store(true)
		} else {
			slog.Error("audio: QueueAudio (converted)", "err", err)
		}
	}
}

// reset discards all buffered audio data and forces a CVT re-probe on the next
// call to play().  Called on server-side audio reset (e.g. seek) so stale audio
// does not keep playing after the stream restarts.
func (a *audioPlayer) reset() {
	if a.deviceID != 0 {
		sdl.ClearQueuedAudio(a.deviceID)
	}
	a.mu.Lock()
	a.cvtKey = [3]int{} // force re-probe
	a.cvtNeed = false
	a.mu.Unlock()
}

func (a *audioPlayer) close() {
	if a.deviceID != 0 {
		sdl.CloseAudioDevice(a.deviceID)
		a.deviceID = 0
	}
}

// reopen closes and reopens the audio device on the calling (main) thread.
// Called after play() signals reopenNeeded due to "Invalid audio device ID".
func (a *audioPlayer) reopen() {
	slog.Warn("audio: device invalid, reopening")
	a.close()
	if err := a.open(); err != nil {
		slog.Error("audio: reopen failed, audio disabled", "err", err)
	}
}

// yuvFrame carries a decoded H.264 frame in NV12 or I420 format from the grdp
// callback to the SDL2 main thread.  buf is the single backing allocation that
// holds all planes; it is returned to yuvBufPool after the texture upload.
// Used only by the fallback path when pre-locking the YUV texture fails.
type yuvFrame struct {
	destX, destY, w, h int
	format             uint32
	y                  []byte
	yStride            int
	u                  []byte
	uStride            int
	v                  []byte
	vStride            int
	uv                 []byte
	uvStride           int
	buf                []byte
	isNull             bool // true when the decoded frame is all-zero (VideoToolbox flush/init artifact)
}

// yuvStage describes the SDL2 YUV texture's pre-locked staging buffer.
// The main goroutine locks the full texture once and publishes the resulting
// yuvStage to H.264 callbacks via yuvStageCh, so callbacks can write decoded
// frames directly into the GPU-accessible unified-memory buffer.
// This halves per-frame data movement: one copy (grdp → MTLBuffer) instead of
// two (grdp → pool → MTLBuffer).
type yuvStage struct {
	all    []byte // entire locked buffer: Y plane then UV/U/V planes
	pitch  int    // row pitch (bytes) in the locked buffer
	tw, th int    // full texture dimensions (not the frame sub-rect)
}

// yuvDone is sent from the H.264 callback goroutine to the main goroutine once
// the decoded frame has been written into the pre-locked yuvStage, signalling
// that Unlock (and the resulting Metal command-buffer commit) is safe to call.
type yuvDone struct {
	destX, destY, w, h int
	isNull             bool // true when the decoded frame is all-zero (VideoToolbox flush/init artifact)
	fullTexture        bool // true when the frame covered the full texture; UV is fully written — skip chroma init on next lock
}

// isNullYUVFrame samples 8 evenly-spaced values from each of the Y and chroma
// planes and returns true only when every sampled value is zero.  In
// limited-range YUV (the standard for H.264) a legitimate black frame has
// Y≥16 and chroma=128, so Y=0 across multiple samples is a reliable marker
// for a null/corrupt decoder output rather than real content.  In full-range
// YUV a black frame has Y=0 but chroma=128, so the chroma check prevents
// false positives.  The O(16) cost is negligible compared to the frame copy.
func isNullYUVFrame(y, chroma []byte) bool {
	ny, nc := len(y), len(chroma)
	if ny == 0 || nc == 0 {
		return false // zero-length slices are handled upstream
	}
	yStep := max(1, ny/8)
	for i := 0; i < ny; i += yStep {
		if y[i] != 0 {
			return false
		}
	}
	cStep := max(1, nc/8)
	for i := 0; i < nc; i += cStep {
		if chroma[i] != 0 {
			return false
		}
	}
	return true
}

func mainLoop(hostPort, domain, user, password string, width, height int, swapAltMeta bool, keyboardType, keyboardLayout string) (err error) {
	cursorCache := make(map[uint16]*sdl.Cursor)
	showCursor := true

	// bitmapBufPool reuses backing arrays for bitmap data copies, reducing
	// GC pressure when many large bitmap updates arrive per second.
	var bitmapBufPool sync.Pool
	// yuvBufPool reuses backing arrays for I420 plane copies (one allocation
	// per frame holds Y+U+V contiguously, ≈3MB at 1920×1080).
	var yuvBufPool sync.Pool

	// reconnecting suppresses the "use of closed network connection" error
	// that the read goroutine emits when Reconnect tears down the old TCP
	// connection.  1 = reconnect in progress, 0 = normal operation.
	var reconnecting atomic.Int32

	// decoderBrokenPending is set by OnDecoderBroken and cleared after the
	// main loop triggers a reconnect.  Using an atomic avoids locking across
	// the callback/main-loop boundary.
	var decoderBrokenPending atomic.Bool

	// connectionErrorPending is set by OnError (TCP-level errors such as
	// "connection reset by peer") and cleared after a successful reconnect.
	// When the video watchdog detects a stall AND this flag is set, the main
	// loop triggers a reconnect instead of just logging — preventing the
	// session from staying black forever after a dropped connection.
	var connectionErrorPending atomic.Bool

	// eventPending prevents redundant SDL user-event pushes when H.264 or
	// bitmap callbacks fire faster than the main loop drains them.  Using
	// CompareAndSwap ensures at most one pending wake-up event sits in the
	// SDL event queue at any time.
	var eventPending atomic.Bool

	if err = sdl.Init(sdl.INIT_VIDEO | sdl.INIT_AUDIO); err != nil {
		return err
	}
	defer sdl.Quit()

	ap := &audioPlayer{}
	if err := ap.open(); err != nil {
		slog.Warn("audio: failed to open device, audio disabled", "err", err)
	}
	defer ap.close()

	sdl.StopTextInput()

	window, err := sdl.CreateWindow("GRDPSDL2", sdl.WINDOWPOS_UNDEFINED,
		sdl.WINDOWPOS_UNDEFINED, int32(width), int32(height), sdl.WINDOW_SHOWN|sdl.WINDOW_RESIZABLE)
	if err != nil {
		return err
	}

	// Pump pending OS events so any initial window-size adjustment (e.g. the
	// OS constraining the window to the available screen area) is delivered
	// before we start the RDP session. This prevents an immediate
	// resize→reconnect on startup.
	sdl.PumpEvents()
	for {
		ev := sdl.PollEvent()
		if ev == nil {
			break
		}
		if we, ok := ev.(*sdl.WindowEvent); ok &&
			(we.Event == sdl.WINDOWEVENT_RESIZED || we.Event == sdl.WINDOWEVENT_SIZE_CHANGED) {
			width = int(we.Data1)
			height = int(we.Data2)
		}
	}
	// Prefer an accelerated renderer with VSync so that renderer.Present()
	// waits for the display vblank.  This caps rendering to the display
	// refresh rate (60/120 Hz), eliminates tearing, and lets the H.264
	// callback write the next frame into the pre-locked MTLBuffer during the
	// vblank stall — pipeline parallelism at no extra cost.
	// Fall back to accelerated without VSync, then software without VSync.
	renderer, err := sdl.CreateRenderer(window, -1, sdl.RENDERER_ACCELERATED|sdl.RENDERER_PRESENTVSYNC)
	if err != nil {
		slog.Warn("vsync renderer unavailable, trying without vsync", "err", err)
		renderer, err = sdl.CreateRenderer(window, -1, sdl.RENDERER_ACCELERATED)
	}
	if err != nil {
		slog.Warn("hardware renderer unavailable, falling back to software", "err", err)
		renderer, err = sdl.CreateRenderer(window, -1, sdl.RENDERER_SOFTWARE)
		if err != nil {
			return err
		}
	}
	defer renderer.Destroy()

	// texture is a BGRA32 streaming texture used for non-H264 bitmap patches
	// (legacy RDP updates, RDPGFX non-AVC codecs, etc.).  It uses BLENDMODE_BLEND
	// so transparent pixels (alpha=0) reveal the H264 IYUV base below.
	// For sessions without H264 the renderer background is black, which shows
	// through transparent pixels, but all real content has alpha=255 so it is fine.
	texture, err := renderer.CreateTexture(uint32(sdl.PIXELFORMAT_BGRA32), sdl.TEXTUREACCESS_STREAMING, int32(width), int32(height))
	if err != nil {
		return err
	}
	defer texture.Destroy()
	texture.SetBlendMode(sdl.BLENDMODE_BLEND)

	// yuvTexture holds the most recent H264 frame as NV12 when supported,
	// otherwise I420 (IYUV). SDL2's renderer uses hardware YUV→RGB shaders,
	// offloading colour conversion entirely from the CPU.
	// On software renderers SDL2 does the conversion in software — no separate
	// GPU/non-GPU code path is needed.
	yuvTextureFormat := uint32(sdl.PIXELFORMAT_IYUV)
	if runtime.GOOS == "darwin" {
		yuvTextureFormat = sdlPixelFormatNV12
	}
	yuvTexture, err := renderer.CreateTexture(yuvTextureFormat, sdl.TEXTUREACCESS_STREAMING, int32(width), int32(height))
	if err != nil && yuvTextureFormat == sdlPixelFormatNV12 {
		slog.Debug("NV12 texture unsupported, trying IYUV", "err", err)
		yuvTextureFormat = uint32(sdl.PIXELFORMAT_IYUV)
		yuvTexture, err = renderer.CreateTexture(yuvTextureFormat, sdl.TEXTUREACCESS_STREAMING, int32(width), int32(height))
	}
	if err != nil {
		// YUV unsupported (unusual but possible on some drivers); fall back
		// to BGRA-only rendering by setting yuvTexture to nil.
		slog.Warn("YUV texture unsupported, H264 will render via BGRA fallback", "err", err)
		yuvTexture = nil
	}
	if yuvTexture != nil {
		defer yuvTexture.Destroy()
	}

	// initYUVBlack writes neutral black (Y=0, chroma=128) into a YUV streaming
	// texture so that any render before the first decoded H.264 frame shows black
	// instead of green.  (Uninitialized NV12/IYUV bytes are typically all-zero;
	// Y=0,U=0,V=0 maps to RGB≈(0,136,0) — the momentary green flash.)
	initYUVBlack := func(tex *sdl.Texture, w, h int, format uint32) {
		yBuf := make([]byte, w*h) // Y=0 (full-range black luma)
		ph := (h + 1) / 2
		if format == sdlPixelFormatNV12 {
			uvBuf := make([]byte, w*ph) // interleaved UV; 128 = neutral chroma
			fillBytes(uvBuf, 128)
			tex.UpdateNV(nil, yBuf, w, uvBuf, w)
		} else {
			// IYUV / I420: separate U and V planes, each half-width.
			hw := (w + 1) / 2
			uvBuf := make([]byte, hw*ph)
			fillBytes(uvBuf, 128)
			tex.UpdateYUV(nil, yBuf, w, uvBuf, hw, uvBuf, hw)
		}
	}
	if yuvTexture != nil {
		initYUVBlack(yuvTexture, width, height, yuvTextureFormat)
	}

	// overlayZero is a pre-zeroed buffer used to reset the overlay texture to
	// fully transparent (BGRA 0,0,0,0) after each H264 full-frame update,
	// ensuring stale non-H264 patches do not obscure the new H264 baseline.
	// Allocated once; reused on every H264 frame and on texture recreation.
	overlayZero := make([]byte, width*height*4)
	// Initialise texture to transparent now so blending is correct from the first frame.
	texture.Update(nil, unsafe.Pointer(&overlayZero[0]), width*4)

	bitmapCh := make(chan []grdp.Bitmap, 32)
	yuvCh := make(chan yuvFrame, 4) // fallback path only (used when pre-lock unavailable)
	yuvReady := false               // true once any H264 frame has been rendered
	yuvTextureIsNull := false       // true when the last uploaded frame was null (Y=0,UV=0=green); skip Copy
	clipboardFromServer := make(chan string, 4)
	clipboardReqCh := make(chan chan string, 1)

	// Pre-lock channels for the primary YUV upload path.
	// yuvStageCh carries the pre-locked staging buffer from the main goroutine
	// to H.264 callbacks; yuvDoneCh signals back when a frame has been written.
	// Capacity 1 each: at most one frame is in flight at any time.
	yuvStageCh := make(chan *yuvStage, 1)
	yuvDoneCh := make(chan yuvDone, 1)
	var yuvWriteWg sync.WaitGroup // counts H.264 writes currently in progress

	// lastH264DropNs records the Unix nanosecond timestamp of the most recent
	// dropped H.264 frame (0 = no recent drop).  Used to set the queueDepth
	// hint that tells the RDP server to reduce H.264 bitrate when SDL's
	// rendering pipeline is congested.
	var lastH264DropNs atomic.Int64

	// preLockYUV locks the full YUV streaming texture and returns a yuvStage
	// that H.264 callbacks can write directly into.  Must be called from the
	// main (SDL) goroutine.  Returns nil if Lock fails (e.g. software renderer).
	// initChroma should be true so that unwritten chroma regions render black
	// (neutral 128) instead of green (Y=0,UV=0 is not a valid black in BT.601).
	// Metal's staging MTLBuffer pool may return a freshly zeroed buffer on each
	// lock, so we cannot rely on previous-frame data in unwritten UV regions.
	preLockYUV := func(tex *sdl.Texture, tw, th int, format uint32, initChroma bool) *yuvStage {
		pixels, pitch, err := tex.Lock(nil)
		if err != nil {
			return nil
		}
		ph := (th + 1) / 2
		yLen := pitch * th
		var all []byte
		if format == sdlPixelFormatNV12 {
			all = unsafe.Slice(&pixels[0], yLen+pitch*ph)
		} else {
			uPitch := (pitch + 1) / 2
			all = unsafe.Slice(&pixels[0], yLen+2*uPitch*ph)
		}
		if initChroma {
			fillBytes(all[yLen:], 128)
		}
		return &yuvStage{all: all, pitch: pitch, tw: tw, th: th}
	}

	// drainPreLock ensures yuvTexture is unlocked before destroying or
	// recreating it.  Waits for any in-progress callback write to finish,
	// then unlocks if the texture is currently held by the pre-lock path.
	// Must be called from the main goroutine.
	drainPreLock := func() {
		yuvWriteWg.Wait() // wait for any concurrent callback write to finish
		select {
		case <-yuvStageCh:
			yuvTexture.Unlock() // stage was pre-locked but callback never consumed it
		default:
			select {
			case <-yuvDoneCh:
				yuvTexture.Unlock() // callback wrote a frame; its Unlock was deferred to us
			default:
				// texture is not currently locked; nothing to do
			}
		}
	}

	// yuvPrimaryPath is true when the pre-lock optimisation is active.
	// It is set to false when Lock fails (e.g. software renderer) so
	// the code automatically degrades to the pool-buffer fallback path.
	yuvPrimaryPath := false
	if yuvTexture != nil {
		if stage := preLockYUV(yuvTexture, width, height, yuvTextureFormat, true); stage != nil {
			yuvPrimaryPath = true
			yuvStageCh <- stage
		}
	}

	// overlayDirtyRects accumulates the rects painted onto the overlay texture
	// (non-H264 bitmap updates) since the last H264 frame.  When the next H264
	// frame arrives we clear only these rects instead of zeroing the entire
	// screen, cutting GPU texture-upload bandwidth significantly.
	// Pre-allocated with capacity 64 to avoid append reallocation on typical frames.
	overlayDirtyRects := make([]sdl.Rect, 0, 64)

	// clearOverlayDirty clears the overlay texture regions accumulated in
	// overlayDirtyRects and resets the slice.  The break-even between individual
	// SDL_UpdateTexture calls (one GPU blit each) and a single full-texture
	// upload is determined by area: if the total dirty area exceeds half the
	// texture, a single upload is cheaper.
	clearOverlayDirty := func() {
		if len(overlayDirtyRects) == 0 {
			return
		}
		var dirtyArea int
		for _, r := range overlayDirtyRects {
			dirtyArea += int(r.W) * int(r.H)
		}
		if dirtyArea*2 > width*height {
			// Batch path: one GPU upload clears the entire overlay texture.
			texture.Update(nil, unsafe.Pointer(&overlayZero[0]), width*4)
		} else {
			for i := range overlayDirtyRects {
				texture.Update(&overlayDirtyRects[i], unsafe.Pointer(&overlayZero[0]), width*4)
			}
		}
		overlayDirtyRects = overlayDirtyRects[:0]
	}

	// lastServerActivity tracks the last time we received any data from the
	// server (bitmap, pointer update, audio, clipboard).  Used by the video
	// stall watchdog below to distinguish a truly stuck stream from an idle
	// remote desktop.  Stored as UnixNano via atomic so it can be updated
	// from network goroutines and read from the main loop without a mutex.
	// Zero means no server activity yet (watchdog disarmed).
	var lastServerActivity atomic.Int64

	// Register a custom SDL event type to wake the main loop when bitmaps arrive.
	bitmapEventType := sdl.RegisterEvents(1)
	// Pre-allocate wake events once so callbacks never heap-allocate on the hot path.
	wakeEvent := &sdl.UserEvent{Type: bitmapEventType}
	decoderBrokenEvent := &sdl.UserEvent{Type: bitmapEventType}

	rdpClient := grdp.NewRdpClient(hostPort, width, height, func(hostPort string) (net.Conn, error) {
		dialer := &net.Dialer{
			KeepAlive: 300 * time.Second,
		}
		conn, err := dialer.Dial("tcp", hostPort)
		if err != nil {
			return nil, err
		}
		if tc, ok := conn.(*net.TCPConn); ok {
			// Disable Nagle's algorithm so keyboard/mouse packets are sent
			// immediately without waiting for more data to accumulate
			// (up to ~40 ms delay otherwise).
			tc.SetNoDelay(true)
			// Increase the TCP receive buffer to 4 MB.  RDP H.264 I-frames can
			// be several hundred KB; a large buffer lets the OS accept a burst
			// without shrinking the receive window and throttling the server,
			// reducing the gap between frames during screen animations.
			tc.SetReadBuffer(4 * 1024 * 1024)
		}
		return conn, nil
	})
	if keyboardType != "" {
		rdpClient.SetKeyboardType(keyboardType)
	}
	if keyboardLayout != "" {
		rdpClient.SetKeyboardLayout(keyboardLayout)
	}
	rdpClient.OnClipboard(
		func(text string) {
			// server → client
			lastServerActivity.Store(time.Now().UnixNano())
			select {
			case clipboardFromServer <- text:
			default:
			}
		},
		func() string {
			// client → server: request clipboard text from main thread
			respCh := make(chan string, 1)
			clipboardReqCh <- respCh
			return <-respCh
		},
	)
	err = rdpClient.Login(domain, user, password)
	if err != nil {
		return err
	}

	rdpClient.OnError(func(e error) {
		if reconnecting.Load() != 0 {
			slog.Debug("on error (during reconnect, suppressed)", "err", e)
			return
		}
		slog.Error("on error", "err", e)
		connectionErrorPending.Store(true)
		if bitmapEventType != sdl.FIRSTEVENT && eventPending.CompareAndSwap(false, true) {
			sdl.PushEvent(wakeEvent)
		}
	}).OnReady(func() {
		slog.Info("on ready")
	}).OnAudio(func(af rdpsnd.AudioFormat, data []byte) {
		lastServerActivity.Store(time.Now().UnixNano())
		ap.play(af, data)
	}).OnAudioReset(func() {
		slog.Debug("audio: reset")
		ap.reset()
	}).OnBitmap(func(bs []grdp.Bitmap) {
		lastServerActivity.Store(time.Now().UnixNano())
		// Bitmap.Data is borrowed from grdp's internal pool; copy it before
		// returning from this callback.  Reuse pooled buffers to avoid
		// allocating fresh backing arrays on every frame.
		for i := range bs {
			src := bs[i].Data
			buf, _ := bitmapBufPool.Get().([]byte)
			if cap(buf) < len(src) {
				buf = make([]byte, len(src))
			} else {
				buf = buf[:len(src)]
			}
			copy(buf, src)
			bs[i].Data = buf
		}
		sent := false
		select {
		case bitmapCh <- bs:
			sent = true
		default:
			// Return buffers to pool since we're dropping this frame.
			for i := range bs {
				bitmapBufPool.Put(bs[i].Data)
			}
			slog.Warn("bitmap channel full, dropping frame")
		}
		// Wake the main loop immediately so it renders without waiting for
		// WaitEventTimeout to expire.
		if sent && bitmapEventType != sdl.FIRSTEVENT && eventPending.CompareAndSwap(false, true) {
			sdl.PushEvent(wakeEvent)
		}
	}).OnPointerHide(func() {
		lastServerActivity.Store(time.Now().UnixNano())
		sdl.ShowCursor(sdl.DISABLE)
		showCursor = false
	}).OnPointerCached(func(idx uint16) {
		lastServerActivity.Store(time.Now().UnixNano())
		if !showCursor {
			sdl.ShowCursor(sdl.ENABLE)
			showCursor = true
		}
		sdl.SetCursor(cursorCache[idx])
	}).OnPointerUpdate(func(idx uint16, bpp uint16, x uint16, y uint16, width uint16, height uint16, mask []byte, data []byte) {
		lastServerActivity.Store(time.Now().UnixNano())
		if !showCursor {
			sdl.ShowCursor(sdl.ENABLE)
			showCursor = true
		}
		if bpp == 24 {
			n := len(data) / 3
			rgba := make([]byte, n*4)
			for i := range n {
				b, g, r := data[3*i], data[3*i+1], data[3*i+2]
				// Branchless alpha: 0x00 when all channels are zero, 0xFF otherwise.
				// (uint(b|g|r) + 255) >> 8 yields 0 for zero input, 1 for any nonzero
				// byte; multiplying by 255 saturates to 0xFF — no branch, no CMOV.
				a := byte(((uint(b|g|r) + 255) >> 8) * 255)
				rgba[4*i], rgba[4*i+1], rgba[4*i+2], rgba[4*i+3] = b, g, r, a
			}
			data = rgba
		}
		surface, err := sdl.CreateRGBSurfaceWithFormatFrom(
			unsafe.Pointer(&data[0]),
			int32(width),
			int32(height),
			32,
			int32(width*4),
			uint32(sdl.PIXELFORMAT_RGBA32),
		)
		if err != nil {
			slog.Error("surface", "err", err)
		}
		defer surface.Free()

		cursor := sdl.CreateColorCursor(surface, int32(x), int32(y))

		if cursor != nil {
			cursorCache[idx] = cursor
			sdl.SetCursor(cursor)
		} else {
			slog.Error("Failed to create cursor")
		}
	}).OnDecoderBroken(func() {
		// All internal recovery attempts (SW fallback + keyframe requests) have
		// been exhausted.  The server will not send a new IDR without a full
		// reconnect; trigger one so the session can resume.
		slog.Warn("decoder broken; scheduling reconnect")
		decoderBrokenPending.Store(true)
		if bitmapEventType != sdl.FIRSTEVENT && eventPending.CompareAndSwap(false, true) {
			sdl.PushEvent(decoderBrokenEvent)
		}
	})

	if yuvTexture != nil {
		if yuvTextureFormat == sdlPixelFormatNV12 {
			rdpClient.OnH264NV12(func(destX, destY, w, h int, y []byte, yStride int, uv []byte, uvStride int) {
				lastServerActivity.Store(time.Now().UnixNano())
				if yuvPrimaryPath {
					// Primary path: write grdp data directly into the pre-locked
					// Metal staging buffer (one copy: grdp → MTLBuffer).
					//
					// Add to WaitGroup BEFORE the select so drainPreLock's Wait()
					// cannot return between the stage receive and the first write,
					// eliminating a narrow use-after-free window during reconnect.
					yuvWriteWg.Add(1)
					select {
					case stage := <-yuvStageCh:
						defer yuvWriteWg.Done()
						ph := (h + 1) / 2
						// UV plane starts after ALL Y rows of the full texture,
						// not just the frame rows.  Using stage.th (texture height)
						// instead of h fixes corruption when the frame is smaller
						// than the texture (e.g. a video window inside the desktop).
						yBaseLen := stage.pitch * stage.th
						uvLen := stage.pitch * ph
						fastPath := stage.pitch == yStride && destX == 0 && destY == 0
						// fullTexture: the frame covers the entire texture, so every
						// Y and UV byte will be overwritten — the next lock can skip
						// the UV pre-initialisation (saves ~1 MB memset per frame).
						fullTexture := fastPath && h == stage.th
						isNull := isNullYUVFrame(y, uv)
						if isNull {
							// Fill UV with 128 (neutral chroma) so Unlock commits black
							// instead of green.  Metal staging buffers may be freshly
							// zeroed (UV=0); Y=0,UV=0 renders as green in BT.601.
							fillBytes(stage.all[yBaseLen:], 128)
						} else if fastPath {
							copy(stage.all[:stage.pitch*h], y[:stage.pitch*h])
							copy(stage.all[yBaseLen:yBaseLen+uvLen], uv[:uvLen])
						} else {
							for row := range h {
								dstOff := (destY+row)*stage.pitch + destX
								copy(stage.all[dstOff:dstOff+w], y[row*yStride:row*yStride+w])
							}
							for row := range ph {
								dstOff := yBaseLen + (destY/2+row)*stage.pitch + destX
								copy(stage.all[dstOff:dstOff+w], uv[row*uvStride:row*uvStride+w])
							}
						}
						done := yuvDone{destX: destX, destY: destY, w: w, h: h,
							isNull: isNull, fullTexture: fullTexture && !isNull}
						select {
						case yuvDoneCh <- done:
						default:
							// Replace stale entry so the main loop always sees the latest frame.
							<-yuvDoneCh
							yuvDoneCh <- done
						}
					default:
						yuvWriteWg.Done()
						slog.Debug("yuv stage not ready, dropping NV12 frame")
						lastH264DropNs.Store(time.Now().UnixNano())
					}
				} else {
					// Fallback path (pre-lock unavailable): copy into pool buffer.
					ph := (h + 1) / 2
					yLen := yStride * h
					uvLen := uvStride * ph
					totalLen := yLen + uvLen
					buf, _ := yuvBufPool.Get().([]byte)
					if cap(buf) < totalLen {
						buf = make([]byte, totalLen)
					} else {
						buf = buf[:totalLen]
					}
					copy(buf[:yLen], y[:yLen])
					copy(buf[yLen:yLen+uvLen], uv[:uvLen])
					frame := yuvFrame{
						destX: destX, destY: destY, w: w, h: h,
						format: yuvTextureFormat,
						y:      buf[:yLen], yStride: yStride,
						uv:     buf[yLen : yLen+uvLen], uvStride: uvStride,
						buf:    buf,
						isNull: isNullYUVFrame(y, uv),
					}
					select {
					case yuvCh <- frame:
					default:
						yuvBufPool.Put(buf)
						slog.Warn("yuv channel full, dropping NV12 frame")
						lastH264DropNs.Store(time.Now().UnixNano())
					}
				}
				if bitmapEventType != sdl.FIRSTEVENT && eventPending.CompareAndSwap(false, true) {
					sdl.PushEvent(wakeEvent)
				}
			})
		} else {
			rdpClient.OnH264I420(func(destX, destY, w, h int, y []byte, yStride int, u []byte, uStride int, v []byte, vStride int) {
				lastServerActivity.Store(time.Now().UnixNano())
				if yuvPrimaryPath {
					// Add to WaitGroup BEFORE the select (same rationale as NV12 path).
					yuvWriteWg.Add(1)
					select {
					case stage := <-yuvStageCh:
						defer yuvWriteWg.Done()
						ph := (h + 1) / 2
						// UV planes start after ALL Y rows of the full texture.
						// Using stage.th (texture height) instead of h fixes
						// corruption when the frame is smaller than the texture.
						yBaseLen := stage.pitch * stage.th
						uPitch := (stage.pitch + 1) / 2
						phTex := (stage.th + 1) / 2
						uvLen := uPitch * ph
						uBaseLen := yBaseLen + uPitch*phTex
						vBaseLen := uBaseLen + uPitch*phTex
						fastPath := stage.pitch == yStride && uPitch == uStride && destX == 0 && destY == 0
						fullTexture := fastPath && h == stage.th
						isNull := isNullYUVFrame(y, u)
						if isNull {
							// Fill U and V planes with 128 so Unlock commits black
							// instead of green.  Metal staging buffers may be freshly
							// zeroed (UV=0); Y=0,UV=0 renders as green in BT.601.
							fillBytes(stage.all[uBaseLen:], 128)
						} else if fastPath {
							copy(stage.all[:stage.pitch*h], y[:stage.pitch*h])
							copy(stage.all[uBaseLen:uBaseLen+uvLen], u[:uvLen])
							copy(stage.all[vBaseLen:vBaseLen+uvLen], v[:uvLen])
						} else {
							w2 := (w + 1) / 2
							for row := range h {
								dstOff := (destY+row)*stage.pitch + destX
								copy(stage.all[dstOff:dstOff+w], y[row*yStride:row*yStride+w])
							}
							for row := range ph {
								dstOff := uBaseLen + (destY/2+row)*uPitch + destX/2
								copy(stage.all[dstOff:dstOff+w2], u[row*uStride:row*uStride+w2])
							}
							for row := range ph {
								dstOff := vBaseLen + (destY/2+row)*uPitch + destX/2
								copy(stage.all[dstOff:dstOff+w2], v[row*vStride:row*vStride+w2])
							}
						}
						done := yuvDone{destX: destX, destY: destY, w: w, h: h,
							isNull: isNull, fullTexture: fullTexture && !isNull}
						select {
						case yuvDoneCh <- done:
						default:
							<-yuvDoneCh
							yuvDoneCh <- done
						}
					default:
						yuvWriteWg.Done()
						slog.Debug("yuv stage not ready, dropping I420 frame")
						lastH264DropNs.Store(time.Now().UnixNano())
					}
				} else {
					ph := (h + 1) / 2
					yLen := yStride * h
					uLen := uStride * ph
					vLen := vStride * ph
					totalLen := yLen + uLen + vLen
					buf, _ := yuvBufPool.Get().([]byte)
					if cap(buf) < totalLen {
						buf = make([]byte, totalLen)
					} else {
						buf = buf[:totalLen]
					}
					copy(buf[:yLen], y[:yLen])
					copy(buf[yLen:yLen+uLen], u[:uLen])
					copy(buf[yLen+uLen:yLen+uLen+vLen], v[:vLen])
					frame := yuvFrame{
						destX: destX, destY: destY, w: w, h: h,
						format: yuvTextureFormat,
						y:      buf[:yLen], yStride: yStride,
						u:      buf[yLen : yLen+uLen], uStride: uStride,
						v:      buf[yLen+uLen : yLen+uLen+vLen], vStride: vStride,
						buf:    buf,
						isNull: isNullYUVFrame(y, u),
					}
					select {
					case yuvCh <- frame:
					default:
						yuvBufPool.Put(buf)
						slog.Warn("yuv channel full, dropping I420 frame")
						lastH264DropNs.Store(time.Now().UnixNano())
					}
				}
				if bitmapEventType != sdl.FIRSTEVENT && eventPending.CompareAndSwap(false, true) {
					sdl.PushEvent(wakeEvent)
				}
			})
		}
	}

	// videoStallTimeout is the maximum duration without ANY response from
	// the server (bitmap, pointer, audio, clipboard) before the session is
	// considered frozen.  An idle remote desktop legitimately sends no
	// frames for long periods, so we must not key this off bitmaps alone.
	//
	// The timeout must be long enough to accommodate the full recovery cycle
	// when the H.264 HW decoder (e.g. VideoToolbox) temporarily produces
	// null frames: grdp's internal freeze threshold (~2 s) + hard reset +
	// IDR request round-trip + server re-encode + first decoded frame.
	// Empirically this cycle can take 5–8 seconds, so 3 s was too short and
	// caused spurious reconnects.  10 s is generous yet still catches a
	// truly stuck session.
	const videoStallTimeout = 10 * time.Second

	// renderDirty counts down from 3 to 0.  It is set to 3 whenever new content
	// arrives (YUV frame, bitmap patch, or reconnect).  The render trio
	// (Clear/Copy/Present) is only called while renderDirty > 0, and is
	// decremented after each Present.  Using 3 instead of 2 is safe for both
	// double- and triple-buffered SDL2 renderers: every backbuffer is refreshed
	// before we pause, preventing a stale buffer from flashing through.
	// Initialized to 3 so the initial black frame is presented correctly.
	renderDirty := 3

	// resetAfterReconnect recreates textures and resets rendering state after a
	// successful Reconnect.  Extracted to avoid duplicating ~25 lines between
	// the resize-reconnect and video-stall-reconnect paths.
	resetAfterReconnect := func(w, h int32) {
		lastServerActivity.Store(0)
		connectionErrorPending.Store(false)
		overlayDirtyRects = overlayDirtyRects[:0]
		texture.Destroy()
		var rerr error
		texture, rerr = renderer.CreateTexture(uint32(sdl.PIXELFORMAT_BGRA32), sdl.TEXTUREACCESS_STREAMING, w, h)
		if rerr != nil {
			slog.Error("CreateTexture after reconnect failed", "err", rerr)
		} else {
			texture.SetBlendMode(sdl.BLENDMODE_BLEND)
			overlayZero = make([]byte, int(w)*int(h)*4)
			texture.Update(nil, unsafe.Pointer(&overlayZero[0]), int(w)*4)
		}
		if yuvTexture != nil {
			drainPreLock()
			yuvTexture.Destroy()
			yuvTexture, rerr = renderer.CreateTexture(yuvTextureFormat, sdl.TEXTUREACCESS_STREAMING, w, h)
			if rerr != nil {
				slog.Warn("IYUV recreate failed after reconnect", "err", rerr)
				yuvTexture = nil
				yuvPrimaryPath = false
			} else {
				initYUVBlack(yuvTexture, int(w), int(h), yuvTextureFormat)
				if stage := preLockYUV(yuvTexture, int(w), int(h), yuvTextureFormat, true); stage != nil {
					yuvPrimaryPath = true
					yuvStageCh <- stage
				} else {
					yuvPrimaryPath = false
				}
			}
		}
		yuvReady = false
		yuvTextureIsNull = false
		ap.reopenNeeded.Store(false)
		ap.reset()
		renderDirty = 3
	}

	quit := false
	var resizePending bool
	var resizeTime time.Time
	var resizeW, resizeH int32

	for !quit {
		// Use a short timeout during active rendering so new frames are picked up
		// quickly.  When idle (renderDirty == 0) extend to 50 ms — H.264 and
		// bitmap callbacks wake the loop via sdl.PushEvent, and SDL input events
		// (keyboard/mouse) also wake it immediately, so responsiveness is unchanged.
		waitMs := 50
		if renderDirty > 0 {
			waitMs = 8
		}
		event := sdl.WaitEventTimeout(waitMs)

		// Coalesce mouse-motion events: only the final position in each tick
		// is sent to the server, eliminating redundant RDP mouse-move packets.
		var mouseMoved bool
		var lastMouseX, lastMouseY int

		for ; event != nil; event = sdl.PollEvent() {
			switch t := event.(type) {
			case *sdl.QuitEvent:
				quit = true

			case *sdl.WindowEvent:
				if t.Event == sdl.WINDOWEVENT_RESIZED {
					resizeW = t.Data1
					resizeH = t.Data2
					resizePending = true
					resizeTime = time.Now()
				}

			case *sdl.KeyboardEvent:
				k := transKey(t.Keysym.Scancode, swapAltMeta)
				if t.State == sdl.RELEASED {
					rdpClient.KeyUp(k)
				} else if t.State == sdl.PRESSED {
					rdpClient.KeyDown(k)
				}

			case *sdl.MouseMotionEvent:
				mouseMoved = true
				lastMouseX, lastMouseY = int(t.X), int(t.Y)

			case *sdl.MouseButtonEvent:
				if t.State == sdl.PRESSED {
					rdpClient.MouseDown(int(t.Button)-1, int(t.X), int(t.Y))
				} else {
					rdpClient.MouseUp(int(t.Button-1), int(t.X), int(t.Y))
				}

			case *sdl.MouseWheelEvent:
				dy := t.PreciseY
				if t.Direction == sdl.MOUSEWHEEL_FLIPPED {
					dy = -dy
				}
				if dy != 0 {
					rdpClient.MouseWheel(float64(dy))
				}

			case *sdl.ClipboardEvent:
				// SDL_CLIPBOARDUPDATE fires whenever the clipboard changes,
				// including when we call sdl.SetClipboardText() ourselves
				// (server → client path).  grdp's suppressNextLocalChange flag
				// absorbs that self-notification, so no echo loop occurs.
				rdpClient.NotifyClipboardChanged()
			}
		}

		if mouseMoved {
			rdpClient.MouseMove(lastMouseX, lastMouseY)
		}

		// Update queueDepth congestion hint every loop iteration so the server
		// reduces H.264 quality while we are dropping frames.  This fires even
		// when all frames are dropped (yuvDoneCh / yuvCh never fire), which is
		// the most important case to signal.
		if dropNs := lastH264DropNs.Load(); dropNs != 0 {
			if time.Since(time.Unix(0, dropNs)) < h264DropCooldown {
				rdpClient.SetQueueDepthHint(h264CongestionHint)
			} else {
				lastH264DropNs.Store(0)
				rdpClient.SetQueueDepthHint(0)
			}
		}

		// Drain incoming bitmaps and update GPU texture on the main thread.
		// Clear the event-pending flag first so the next callback push is not suppressed.
		eventPending.Store(false)

		// Process H.264 YUV frames.
		if yuvPrimaryPath {
			// Primary path: the callback wrote directly into the pre-locked Metal
			// staging buffer.  We just need to Unlock (which commits one Metal
			// command buffer to blit the buffer into the YUV texture) and
			// immediately re-lock so the staging buffer is ready for the next frame.
			select {
			case done := <-yuvDoneCh:
				yuvTexture.Unlock() // GPU upload: grdp → MTLBuffer already done by callback
				// Do NOT clear the overlay on null frames: null frames must not erase bitmap patches.
				if !done.isNull {
					clearOverlayDirty()
				}
				// Re-lock immediately so the next callback can write without waiting.
				// Skip UV pre-initialisation when the last frame was full-texture:
				// every UV byte was already overwritten by the callback, so Metal's
				// staging buffer (even if freshly zeroed by the pool) has correct 4:2:0
				// chroma.  For partial frames the init is still required to avoid green
				// fringes (UV=0 ≈ RGB(0,136,0) in BT.601) in unwritten regions.
				// After a null frame, always reinit (chroma must be set to 128).
				needsChromaInit := !done.fullTexture || done.isNull
				if stage := preLockYUV(yuvTexture, width, height, yuvTextureFormat, needsChromaInit); stage != nil {
					select {
					case yuvStageCh <- stage:
					default:
						// Channel already has a stage (shouldn't happen); release this one.
						yuvTexture.Unlock()
					}
				} else {
					slog.Warn("YUV pre-lock failed after frame, switching to fallback path")
					yuvPrimaryPath = false
				}
				if !done.isNull {
					// Null frames (VideoToolbox flush/init artifacts: Y=0,UV=0) are
					// unlocked above to keep the pipeline moving, but not rendered —
					// showing them would flash green for one display frame.
					yuvReady = true
					yuvTextureIsNull = false
				} else {
					yuvTextureIsNull = true
				}
				renderDirty = 3
			default:
			}
		} else {
			// Fallback path: drain pool-buffer frames copied by the callback and
			// upload them with the Lock/copy/Unlock path (two copies total).
			var latestYUV yuvFrame
			haveYUV := false
		drainYUV:
			for {
				select {
				case frame := <-yuvCh:
					if haveYUV {
						yuvBufPool.Put(latestYUV.buf)
					}
					latestYUV = frame
					haveYUV = true
				default:
					break drainYUV
				}
			}
			if haveYUV {
				if latestYUV.isNull {
					// Null frame (Y=0,UV=0): discard — uploading green data would flash
					// through non-bitmap areas.  Keep the previous real frame visible.
					yuvBufPool.Put(latestYUV.buf)
					yuvTextureIsNull = true
				} else {
					clearOverlayDirty()
					rect := sdl.Rect{X: int32(latestYUV.destX), Y: int32(latestYUV.destY), W: int32(latestYUV.w), H: int32(latestYUV.h)}
					uploadYUVFrame(latestYUV, yuvTexture, &rect)
					yuvBufPool.Put(latestYUV.buf)
					yuvReady = true
					yuvTextureIsNull = false
				}
				renderDirty = 3
			}
		}

	// Drain at most maxBitmapBatchesPerTick batches per render tick so that
	// a burst of updates from the server (e.g. VirtualBox) does not accumulate
	// all GPU texture writes into a single blocking spike.  Any remaining
	// batches stay in bitmapCh and are processed in the next iteration.
	for range maxBitmapBatchesPerTick {
		select {
		case bs := <-bitmapCh:
			paintImages(bs, texture, &overlayDirtyRects)
			for i := range bs {
				bitmapBufPool.Put(bs[i].Data)
			}
			renderDirty = 3
		default:
		}
	}
		// Render only when content has changed.  renderDirty counts down so that
		// every backbuffer is refreshed before we pause (SDL2 double/triple
		// buffering requires re-issuing the same draw commands until all buffers
		// are updated, otherwise a stale empty backbuffer flashes through).
		if renderDirty > 0 {
			// Clear to black first so the uninitialized renderer background (which
			// shows through the initially-transparent BGRA overlay) is not visible.
			renderer.SetDrawColor(0, 0, 0, 255)
			renderer.Clear()
			if yuvReady && !yuvTextureIsNull {
				renderer.Copy(yuvTexture, nil, nil)
			}
			// Skip the overlay copy when the texture is fully transparent —
			// clearOverlayDirty() has already zeroed every dirty rect, so
			// overlayDirtyRects being empty means there is no bitmap content to
			// blend.  In bitmap-only sessions (!yuvReady) we always copy because
			// the overlay holds all visible content.
			if len(overlayDirtyRects) > 0 || !yuvReady {
				renderer.Copy(texture, nil, nil)
			}
			renderer.Present()
			renderDirty--
		}

		// Handle clipboard from server (server → client).
		// sdl.SetClipboardText() fires SDL_CLIPBOARDUPDATE, which the event loop
		// above handles via NotifyClipboardChanged(); grdp's suppress flag
		// prevents the echo from being forwarded back to the server.
		select {
		case text := <-clipboardFromServer:
			sdl.SetClipboardText(text)
		default:
		}

		// Handle clipboard request from server (client → server)
		select {
		case respCh := <-clipboardReqCh:
			text, _ := sdl.GetClipboardText()
			respCh <- text
		default:
		}

		if resizePending && time.Since(resizeTime) > 500*time.Millisecond {
			resizePending = false
			slog.Info("Window resized, reconnecting", "width", resizeW, "height", resizeH)
			reconnecting.Store(1)
			reconnErr := rdpClient.Reconnect(int(resizeW), int(resizeH))
			reconnecting.Store(0)
			if reconnErr != nil {
				slog.Error("Reconnect failed", "err", reconnErr)
			} else {
				resetAfterReconnect(resizeW, resizeH)
			}
		}

		if decoderBrokenPending.CompareAndSwap(true, false) && !resizePending {
			w, h := window.GetSize()
			slog.Info("Decoder broken, reconnecting", "width", w, "height", h)
			reconnecting.Store(1)
			reconnErr := rdpClient.Reconnect(int(w), int(h))
			reconnecting.Store(0)
			if reconnErr != nil {
				slog.Error("Reconnect (decoder broken) failed", "err", reconnErr)
			} else {
				resetAfterReconnect(w, h)
			}
		}

		// Video watchdog: reconnect if a connection error was reported and the
		// session has been silent for videoStallTimeout; otherwise just log
		// (the remote desktop may legitimately be idle for long periods).
		lastNS := lastServerActivity.Load()
		if lastNS != 0 && !resizePending {
			elapsed := time.Since(time.Unix(0, lastNS))
			if elapsed > videoStallTimeout {
				if connectionErrorPending.CompareAndSwap(true, false) {
					w, h := window.GetSize()
					slog.Warn("Video stalled after connection error, reconnecting",
						"stalled", elapsed.Round(time.Millisecond), "width", w, "height", h)
					reconnecting.Store(1)
					reconnErr := rdpClient.Reconnect(int(w), int(h))
					reconnecting.Store(0)
					if reconnErr != nil {
						slog.Error("Reconnect (stall) failed", "err", reconnErr)
						connectionErrorPending.Store(true) // retry next stall cycle
					} else {
						resetAfterReconnect(w, h)
					}
				} else {
					slog.Warn("Video stalled", "stalled", elapsed.Round(time.Millisecond))
					lastServerActivity.Store(time.Now().UnixNano()) // reset to avoid repeated log spam
				}
			}
		}
		// Audio device recovery: play() sets reopenNeeded when SDL2 reports
		// "Invalid audio device ID" (macOS Core Audio sometimes invalidates the
		// device after many reconnects).  Reopen on the main thread to satisfy
		// SDL2/Core Audio threading requirements.
		if ap.reopenNeeded.CompareAndSwap(true, false) {
			ap.reopen()
		}
	}

	err = window.Destroy()
	return err
}

func transKey(scancode sdl.Scancode, transAltMeta bool) int {
	if transAltMeta {
		if scancode == 0xE2 || scancode == 0xe6 {
			scancode += 1
		} else if scancode == 0xe3 || scancode == 0xE7 {
			scancode -= 1
		}
	}

	if v, ok := scancodeMap[scancode]; ok {
		return v
	}
	return 0
}

var scancodeMap = map[sdl.Scancode]int{
	sdl.SCANCODE_UNKNOWN:      0x0000,
	sdl.SCANCODE_ESCAPE:       0x0001,
	sdl.SCANCODE_1:            0x0002,
	sdl.SCANCODE_2:            0x0003,
	sdl.SCANCODE_3:            0x0004,
	sdl.SCANCODE_4:            0x0005,
	sdl.SCANCODE_5:            0x0006,
	sdl.SCANCODE_6:            0x0007,
	sdl.SCANCODE_7:            0x0008,
	sdl.SCANCODE_8:            0x0009,
	sdl.SCANCODE_9:            0x000A,
	sdl.SCANCODE_0:            0x000B,
	sdl.SCANCODE_MINUS:        0x000C,
	sdl.SCANCODE_EQUALS:       0x000D,
	sdl.SCANCODE_BACKSPACE:    0x000E,
	sdl.SCANCODE_TAB:          0x000F,
	sdl.SCANCODE_Q:            0x0010,
	sdl.SCANCODE_W:            0x0011,
	sdl.SCANCODE_E:            0x0012,
	sdl.SCANCODE_R:            0x0013,
	sdl.SCANCODE_T:            0x0014,
	sdl.SCANCODE_Y:            0x0015,
	sdl.SCANCODE_U:            0x0016,
	sdl.SCANCODE_I:            0x0017,
	sdl.SCANCODE_O:            0x0018,
	sdl.SCANCODE_P:            0x0019,
	sdl.SCANCODE_LEFTBRACKET:  0x001A,
	sdl.SCANCODE_RIGHTBRACKET: 0x001B,
	sdl.SCANCODE_RETURN:       0x001C,
	sdl.SCANCODE_LCTRL:        0x001D,
	sdl.SCANCODE_A:            0x001E,
	sdl.SCANCODE_S:            0x001F,
	sdl.SCANCODE_D:            0x0020,
	sdl.SCANCODE_F:            0x0021,
	sdl.SCANCODE_G:            0x0022,
	sdl.SCANCODE_H:            0x0023,
	sdl.SCANCODE_J:            0x0024,
	sdl.SCANCODE_K:            0x0025,
	sdl.SCANCODE_L:            0x0026,
	sdl.SCANCODE_SEMICOLON:    0x0027,
	sdl.SCANCODE_APOSTROPHE:   0x0028,
	sdl.SCANCODE_GRAVE:        0x0029,
	sdl.SCANCODE_LSHIFT:       0x002A,
	sdl.SCANCODE_BACKSLASH:    0x002B,
	sdl.SCANCODE_Z:            0x002C,
	sdl.SCANCODE_X:            0x002D,
	sdl.SCANCODE_C:            0x002E,
	sdl.SCANCODE_V:            0x002F,
	sdl.SCANCODE_B:            0x0030,
	sdl.SCANCODE_N:            0x0031,
	sdl.SCANCODE_M:            0x0032,
	sdl.SCANCODE_COMMA:        0x0033,
	sdl.SCANCODE_PERIOD:       0x0034,
	sdl.SCANCODE_SLASH:        0x0035,
	sdl.SCANCODE_RSHIFT:       0x0036,
	sdl.SCANCODE_KP_MULTIPLY:  0x0037,
	sdl.SCANCODE_LALT:         0x0038,
	sdl.SCANCODE_SPACE:        0x0039,
	sdl.SCANCODE_CAPSLOCK:     0x003A,
	sdl.SCANCODE_F1:           0x003B,
	sdl.SCANCODE_F2:           0x003C,
	sdl.SCANCODE_F3:           0x003D,
	sdl.SCANCODE_F4:           0x003E,
	sdl.SCANCODE_F5:           0x003F,
	sdl.SCANCODE_F6:           0x0040,
	sdl.SCANCODE_F7:           0x0041,
	sdl.SCANCODE_F8:           0x0042,
	sdl.SCANCODE_F9:           0x0043,
	sdl.SCANCODE_F10:          0x0044,
	// sdl.SCANCODE_PAUSE:        0x0045,
	sdl.SCANCODE_SCROLLLOCK:   0x0046,
	sdl.SCANCODE_KP_7:         0x0047,
	sdl.SCANCODE_KP_8:         0x0048,
	sdl.SCANCODE_KP_9:         0x0049,
	sdl.SCANCODE_KP_MINUS:     0x004A,
	sdl.SCANCODE_KP_4:         0x004B,
	sdl.SCANCODE_KP_5:         0x004C,
	sdl.SCANCODE_KP_6:         0x004D,
	sdl.SCANCODE_KP_PLUS:      0x004E,
	sdl.SCANCODE_KP_1:         0x004F,
	sdl.SCANCODE_KP_2:         0x0050,
	sdl.SCANCODE_KP_3:         0x0051,
	sdl.SCANCODE_KP_0:         0x0052,
	sdl.SCANCODE_KP_DECIMAL:   0x0053,
	sdl.SCANCODE_F11:          0x0057,
	sdl.SCANCODE_F12:          0x0058,
	sdl.SCANCODE_KP_EQUALS:    0x0059,
	sdl.SCANCODE_KP_ENTER:     0xE01C,
	sdl.SCANCODE_RCTRL:        0xE01D,
	sdl.SCANCODE_KP_DIVIDE:    0xE035,
	sdl.SCANCODE_PRINTSCREEN:  0xE037,
	sdl.SCANCODE_RALT:         0xE038,
	sdl.SCANCODE_NUMLOCKCLEAR: 0xE045,
	sdl.SCANCODE_PAUSE:        0xE046,
	sdl.SCANCODE_HOME:         0xE047,
	sdl.SCANCODE_UP:           0xE048,
	sdl.SCANCODE_PAGEUP:       0xE049,
	sdl.SCANCODE_LEFT:         0xE04B,
	sdl.SCANCODE_RIGHT:        0xE04D,
	sdl.SCANCODE_END:          0xE04F,
	sdl.SCANCODE_DOWN:         0xE050,
	sdl.SCANCODE_PAGEDOWN:     0xE051,
	sdl.SCANCODE_INSERT:       0xE052,
	sdl.SCANCODE_DELETE:       0xE053,
	sdl.SCANCODE_MENU:         0xE05D,
}

func main() {
	// LockOSThread pins the main goroutine to the OS thread for the lifetime
	// of the process.  SDL2 on macOS requires all rendering and event calls
	// to originate from the main OS thread (Cocoa / NSApplication constraint).
	// Without this the Go scheduler may migrate the goroutine to a different
	// thread, causing subtle crashes or missing events on macOS.
	runtime.LockOSThread()

	// handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})
	// slog.SetDefault(slog.New(handler))

	swapAltMeta := flag.Bool("swap-alt-meta", false, "swap alt and meta key")
	debugLog := flag.Bool("debug", false, "enable debug logging")
	flag.Parse()

	if *debugLog {
		handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})
		slog.SetDefault(slog.New(handler))
	}
	slog.Debug("flag", "swap_alt_meta", *swapAltMeta, "debug", *debugLog)

	hostPort := strings.Join([]string{os.Getenv("GRDP_HOST"), os.Getenv("GRDP_PORT")}, ":")
	domain := os.Getenv("GRDP_DOMAIN")
	user := os.Getenv("GRDP_USER")
	password := os.Getenv("GRDP_PASSWORD")
	keyboardType := os.Getenv("GRDP_KEYBOARD_TYPE")
	keyboardLayout := os.Getenv("GRDP_KEYBOARD_LAYOUT")
	var width, height int
	_, err := fmt.Sscanf(os.Getenv("GRDP_WINDOW_SIZE"), "%dx%d", &width, &height)
	if err != nil {
		width, height = 1280, 800
	}

	mainLoop(hostPort, domain, user, password, width, height, *swapAltMeta, keyboardType, keyboardLayout)
}
