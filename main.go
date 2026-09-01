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

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

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

// ensureOpaqueBGRA sets the alpha byte (offset +3) of every 32-bit pixel in
// data to 0xFF (opaque). RDP 32bpp bitmaps (XRGB/BGRX) typically carry 0x00 in
// the alpha byte; because the overlay texture uses SDL_BLENDMODE_BLEND, an
// alpha of 0x00 renders as completely transparent and lets the underlying
// YUV plane show through, causing green/black artifacts on mouse hover/movement.
func ensureOpaqueBGRA(data []byte) {
	n := len(data) / 4
	for i := range n {
		data[i*4+3] = 0xFF
	}
}

// paintImages uploads each bitmap patch into the SDL2 streaming texture.
// Dirty rects are appended to dirtyRects so the caller can later clear only
// those regions (instead of the entire texture) when a new H.264 frame arrives.
//
// Slow-path (legacy bit-depth) bitmaps are uncommon; a shared BGRA scratch
// buffer is reused across those patches to avoid per-patch allocations.
// Fast-path (BitsPerPixel==4, native BGRA) bitmaps use one SDL_UpdateTexture
// call per patch.  SDL_LockTexture on macOS/Metal returns a zero-initialised
// staging buffer rather than existing texture content, so batching patches into
// a single Lock+Unlock would paint black over any gap; per-patch UpdateTexture
// avoids this.  Both cases are handled in a single loop over bs.
func paintImages(bs []grdp.Bitmap, texture *sdl.Texture, width, height int, dirtyRects *[]sdl.Rect) {
	var bgraBuf []byte
	for _, bm := range bs {
		if bm.BitsPerPixel != 4 {
			// Slow path: convert legacy bit-depth to BGRA32, reusing bgraBuf.
			bgraBuf = bm.FillBGRA(bgraBuf)
			if len(bgraBuf) == 0 {
				continue
			}
			ensureOpaqueBGRA(bgraBuf)
			w := min(bm.DestRight-bm.DestLeft+1, bm.Width)
			h := min(bm.DestBottom-bm.DestTop+1, bm.Height)
			rect := sdl.Rect{X: int32(bm.DestLeft), Y: int32(bm.DestTop), W: int32(w), H: int32(h)}
			texture.Update(&rect, unsafe.Pointer(&bgraBuf[0]), bm.Width*4)
			*dirtyRects = append(*dirtyRects, rect)
			continue
		}

		// Fast path: native BGRA — one SDL_UpdateTexture per patch.
		if len(bm.Data) == 0 {
			continue
		}
		ensureOpaqueBGRA(bm.Data)
		w := min(bm.DestRight-bm.DestLeft+1, bm.Width)
		h := min(bm.DestBottom-bm.DestTop+1, bm.Height)
		if w <= 0 || h <= 0 {
			continue
		}
		x0 := max(bm.DestLeft, 0)
		y0 := max(bm.DestTop, 0)
		x1 := min(bm.DestLeft+w, width)
		y1 := min(bm.DestTop+h, height)
		if x1 <= x0 || y1 <= y0 {
			continue
		}
		srcCol := x0 - bm.DestLeft
		srcRow := y0 - bm.DestTop
		srcOff := srcRow*bm.Width*4 + srcCol*4
		rect := sdl.Rect{X: int32(x0), Y: int32(y0), W: int32(x1 - x0), H: int32(y1 - y0)}
		texture.Update(&rect, unsafe.Pointer(&bm.Data[srcOff]), bm.Width*4)
		*dirtyRects = append(*dirtyRects, rect)
	}
}

// isInvalidChromaNV12 checks if an NV12 frame has corrupted/zero chroma.
// In valid YCbCr, neutral/grayscale chroma is ~128. Values near 0 (U<30 && V<30)
// indicate uninitialized/corrupt buffers that render as bright green ("green curtain").
func isInvalidChromaNV12(uv []byte, uvStride, w, h int) bool {
	if len(uv) == 0 || w <= 0 || h <= 0 {
		return true
	}
	ph := (h + 1) / 2
	pw := (w + 1) / 2
	xs := [3]int{pw / 4, pw / 2, 3 * pw / 4}
	ys := [3]int{ph / 4, ph / 2, 3 * ph / 4}
	zeroCount := 0
	for _, y := range ys {
		rowOff := y * uvStride
		for _, x := range xs {
			off := rowOff + x*2
			if off+1 < len(uv) {
				u := uv[off]
				v := uv[off+1]
				if u < 30 && v < 30 {
					zeroCount++
				}
			}
		}
	}
	return zeroCount >= 3
}

// isInvalidChromaI420 checks if an I420 frame has corrupted/zero chroma.
func isInvalidChromaI420(u, v []byte, uStride, vStride, w, h int) bool {
	if len(u) == 0 || len(v) == 0 || w <= 0 || h <= 0 {
		return true
	}
	ph := (h + 1) / 2
	pw := (w + 1) / 2
	xs := [3]int{pw / 4, pw / 2, 3 * pw / 4}
	ys := [3]int{ph / 4, ph / 2, 3 * ph / 4}
	zeroCount := 0
	for _, y := range ys {
		uOff := y * uStride
		vOff := y * vStride
		for _, x := range xs {
			if uOff+x < len(u) && vOff+x < len(v) {
				uVal := u[uOff+x]
				vVal := v[vOff+x]
				if uVal < 30 && vVal < 30 {
					zeroCount++
				}
			}
		}
	}
	return zeroCount >= 3
}

// uploadYUVFrame uploads a decoded H.264 YUV frame into the SDL2 YUV texture.
func uploadYUVFrame(frame yuvFrame, texture *sdl.Texture, rect *sdl.Rect) {
	if frame.format == sdlPixelFormatNV12 {
		texture.UpdateNV(rect, frame.y, frame.yStride, frame.uv, frame.uvStride)
	} else {
		texture.UpdateYUV(rect, frame.y, frame.yStride, frame.u, frame.uStride, frame.v, frame.vStride)
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
	reopenNeeded atomic.Bool // set from play() on "Invalid audio device ID"; cleared by reopen() on main thread
	mu           sync.Mutex  // protects cvtKey, cvtNeed, cvtBuf
	cvtKey       [3]int      // [SamplesPerSec, Channels, BitsPerSample] of the last-probed CVT
	cvtNeed      bool        // true when the last-probed CVT actually transforms data
	cvtBuf       []byte      // reusable scratch buffer for in-place conversion
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
}

func mainLoop(hostPort, domain, user, password string, width, height int, swapAltMeta bool, keyboardType, keyboardLayout string, disableAVC444 bool) (err error) {
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
	//
	// Force BT.709 explicitly so SDL2 uses the same colour matrix as the AVC444
	// LC=2 BGRA combine path for all resolutions.  Without this, SDL2 auto-
	// selects BT.601 for SD (≤1000×600) sessions and BT.709 for HD sessions,
	// which would cause a coefficient mismatch for low-resolution RDP targets.
	sdl.SetYUVConversionMode(sdl.YUV_CONVERSION_BT709)
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

	bitmapCh := make(chan []grdp.Bitmap, 64)
	yuvCh := make(chan yuvFrame, 32)
	yuvReady := false // true once any H264 frame has been rendered
	clipboardFromServer := make(chan string, 4)
	clipboardReqCh := make(chan chan string, 1)

	// lastH264DropNs records the Unix nanosecond timestamp of the most recent
	// dropped H.264 frame (0 = no recent drop).  Used to set the queueDepth
	// hint that tells the RDP server to reduce H.264 bitrate when SDL's
	// rendering pipeline is congested.
	var lastH264DropNs atomic.Int64

	// overlayDirtyRects accumulates the rects painted onto the overlay texture
	// (non-H264 bitmap updates) since the last H264 frame.  When the next H264
	// frame arrives we clear only these rects instead of zeroing the entire
	// screen, cutting GPU texture-upload bandwidth significantly.
	// Pre-allocated with capacity 64 to avoid append reallocation on typical frames.
	overlayDirtyRects := make([]sdl.Rect, 0, 64)

	// overlayHasContent is true when the overlay (BGRA) texture contains at least
	// one non-transparent bitmap patch.  It is set to true whenever paintImages
	// writes any pixels and reset to false only when a Tier-1 clearOverlayDirty
	// wipes the entire texture or on reconnect.  Unlike overlayDirtyRects (which
	// is emptied on every H.264 frame), this flag persists across frames so that
	// persistent UI elements (desktop background, taskbar, menus) outside the
	// recently-dirtied video region are still copied to the renderer even after
	// clearOverlayDirty empties overlayDirtyRects.
	overlayHasContent := false

	// clearOverlayDirty clears the entire overlay texture to transparent and resets state.
	clearOverlayDirty := func() {
		if !overlayHasContent && len(overlayDirtyRects) == 0 {
			return
		}
		texture.Update(nil, unsafe.Pointer(&overlayZero[0]), width*4)
		overlayHasContent = false
		overlayDirtyRects = overlayDirtyRects[:0]
	}

	// lastServerActivity tracks the last time we received any data from the
	// server (bitmap, pointer update, audio, clipboard).  Used by the video
	// stall watchdog below to distinguish a truly stuck stream from an idle
	// remote desktop.  Stored as UnixNano via atomic so it can be updated
	// from network goroutines and read from the main loop without a mutex.
	// Zero means no server activity yet (watchdog disarmed).
	var lastServerActivity atomic.Int64

	// everShowedFrame is set once a genuine (non-null, non-dropped) video
	// frame has been rendered at least once.  The video-stall watchdog below
	// normally assumes a silent stream just means an idle remote desktop and
	// does not reconnect — but that assumption only holds once the session
	// has actually shown something.  If the server goes silent for
	// videoStallTimeout before we have ever shown a real frame, the session
	// is stuck (e.g. the decoder never recovered and the server also stopped
	// responding), not idle, so the watchdog should reconnect unconditionally.
	var everShowedFrame atomic.Bool

	// neverShownConnectNs marks when the current connection attempt began
	// (set at the initial connect and reset after every successful
	// reconnect).
	neverShownConnectNs := time.Now().UnixNano()
	neverShownLastKeyframeNs := int64(0)

	// lastYUVFrameTime records the main-loop time when the most recent
	// H.264 YUV frame was processed (including null frames that are dropped
	// before rendering).  Used by the SW fallback BGRA watchdog to detect
	// when the hardware decoder has stopped delivering YUV frames.  Zero
	// means no YUV frame has been processed yet.
	var lastYUVFrameTime atomic.Int64
	// fullScreenBitmapStartTime records when the current streak of full-screen
	// BGRA bitmap patches began.  Zero means no streak is active.
	var fullScreenBitmapStartTime atomic.Int64
	// lastFullScreenBitmapTime records the most recent full-screen BGRA
	// bitmap patch.  Used to ensure the streak is still alive before
	// triggering a reconnect.
	var lastFullScreenBitmapTime atomic.Int64

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
	if disableAVC444 {
		rdpClient.DisableAVC444()
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
		// Use CompareAndSwap so only the first error is logged; cascading errors
		// (e.g. "use of closed network connection" after a server disconnect) are
		// suppressed to avoid redundant noise.
		if !connectionErrorPending.CompareAndSwap(false, true) {
			slog.Debug("on error (cascaded, suppressed)", "err", e)
			return
		}
		slog.Warn("on error", "err", e)
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
		if len(bs) > 0 {
			everShowedFrame.Store(true)
		}
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
				everShowedFrame.Store(true)
				lastServerActivity.Store(time.Now().UnixNano())

				if isInvalidChromaNV12(uv, uvStride, w, h) {
					slog.Debug("dropping invalid chroma NV12 frame (green guard)")
					return
				}

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
					format:   yuvTextureFormat,
					y:        buf[:yLen],
					yStride:  yStride,
					uv:       buf[yLen : yLen+uvLen],
					uvStride: uvStride,
					buf:      buf,
				}
				select {
				case yuvCh <- frame:
				default:
					select {
					case old := <-yuvCh:
						yuvBufPool.Put(old.buf)
					default:
					}
					select {
					case yuvCh <- frame:
					default:
						yuvBufPool.Put(buf)
					}
				}
				if bitmapEventType != sdl.FIRSTEVENT && eventPending.CompareAndSwap(false, true) {
					sdl.PushEvent(wakeEvent)
				}
			})
		} else {
			rdpClient.OnH264I420(func(destX, destY, w, h int, y []byte, yStride int, u []byte, uStride int, v []byte, vStride int) {
				everShowedFrame.Store(true)
				lastServerActivity.Store(time.Now().UnixNano())

				if isInvalidChromaI420(u, v, uStride, vStride, w, h) {
					slog.Debug("dropping invalid chroma I420 frame (green guard)")
					return
				}

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
					format:  yuvTextureFormat,
					y:       buf[:yLen],
					yStride: yStride,
					u:       buf[yLen : yLen+uLen],
					uStride: uStride,
					v:       buf[yLen+uLen : yLen+uLen+vLen],
					vStride: vStride,
					buf:     buf,
				}
				select {
				case yuvCh <- frame:
				default:
					select {
					case old := <-yuvCh:
						yuvBufPool.Put(old.buf)
					default:
					}
					select {
					case yuvCh <- frame:
					default:
						yuvBufPool.Put(buf)
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

	// neverShownKeyframeGrace is how long into a still-never-shown-a-frame
	// connection attempt we wait before sending a single ForceRefresh
	// (RequestKeyframe) request, ahead of the full videoStallTimeout
	// reconnect.  Traffic capture shows the server sends roughly half a
	// second of black warm-up frames and then goes completely silent (no
	// traffic on any channel, not just video) for several seconds — the
	// warm-up burst is reliably over well before this grace elapses, so a
	// single request here does not race it (an earlier attempt sent 4
	// requests every 2s starting at 1.5s, which did race the burst and
	// produced visibly corrupted frames).  Must be comfortably less than
	// videoStallTimeout to leave time for the server to respond before the
	// reconnect fires.
	const neverShownKeyframeGrace = 2500 * time.Millisecond
	const neverShownKeyframeInterval = 2 * time.Second

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
		width, height = int(w), int(h)
		lastServerActivity.Store(0)
		neverShownConnectNs = time.Now().UnixNano()
		neverShownLastKeyframeNs = 0
		connectionErrorPending.Store(false)
		overlayDirtyRects = overlayDirtyRects[:0]
		overlayHasContent = false
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
			yuvTexture.Destroy()
			yuvTexture, rerr = renderer.CreateTexture(yuvTextureFormat, sdl.TEXTUREACCESS_STREAMING, w, h)
			if rerr != nil {
				slog.Warn("IYUV recreate failed after reconnect", "err", rerr)
				yuvTexture = nil
			} else {
				initYUVBlack(yuvTexture, int(w), int(h), yuvTextureFormat)
			}
		}
		yuvReady = false
		lastYUVFrameTime.Store(0)
		fullScreenBitmapStartTime.Store(0)
		lastFullScreenBitmapTime.Store(0)
		ap.reopenNeeded.Store(false)
		ap.reset()
		renderDirty = 3
	}

	quit := false
	var resizePending bool
	var resizeTime time.Time
	var resizeW, resizeH int32
	// allBitmaps accumulates bitmaps drained from bitmapCh each render tick.
	// Declared outside the loop so the backing array is reused across ticks.
	var allBitmaps []grdp.Bitmap

	for !quit {
		// Always use a 50 ms timeout.  H.264 and bitmap callbacks push a wake
		// event via sdl.PushEvent, so WaitEventTimeout returns near-immediately
		// when new content arrives regardless of the timeout value.
		// The short 8 ms polling previously used during renderDirty > 0 was
		// redundant: wake events already drive timely rendering, and the
		// extra CPU burn from 125 Hz busy-polling was the main idle CPU cost.
		event := sdl.WaitEventTimeout(50)

		// Coalesce mouse-motion events: only the final position in each tick
		// is sent to the server, eliminating redundant RDP mouse-move packets.
		var mouseMoved bool
		var lastMouseX, lastMouseY int

		for ; event != nil; event = sdl.PollEvent() {
			switch t := event.(type) {
			case *sdl.QuitEvent:
				quit = true

			case *sdl.WindowEvent:
				if t.Event == sdl.WINDOWEVENT_RESIZED || t.Event == sdl.WINDOWEVENT_SIZE_CHANGED {
					dw := abs(int(t.Data1) - width)
					dh := abs(int(t.Data2) - height)
					if dw > 2 || dh > 2 {
						resizeW = t.Data1
						resizeH = t.Data2
						resizePending = true
						resizeTime = time.Now()
					}
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
					rdpClient.MouseUp(int(t.Button)-1, int(t.X), int(t.Y))
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

		// Snapshot current time once for all time-based checks in this iteration,
		// avoiding redundant time.Now() syscalls (congestion hint, resize debounce,
		// stall watchdog — up to 4 calls reduced to 1).
		now := time.Now()
		nowNs := now.UnixNano()

		// Update queueDepth congestion hint every loop iteration so the server
		// reduces H.264 quality while we are dropping frames.  This fires even
		// when all frames are dropped (yuvDoneCh / yuvCh never fire), which is
		// the most important case to signal.
		if dropNs := lastH264DropNs.Load(); dropNs != 0 {
			if nowNs-dropNs < int64(h264DropCooldown) {
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
		if yuvTexture != nil {
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
				isFullScreen := latestYUV.destX == 0 && latestYUV.destY == 0 &&
					latestYUV.w >= width && latestYUV.h >= height
				if isFullScreen {
					clearOverlayDirty()
				}
				rectW := min(latestYUV.w, width-latestYUV.destX)
				rectH := min(latestYUV.h, height-latestYUV.destY)
				if rectW > 0 && rectH > 0 {
					rect := sdl.Rect{X: int32(latestYUV.destX), Y: int32(latestYUV.destY), W: int32(rectW), H: int32(rectH)}
					uploadYUVFrame(latestYUV, yuvTexture, &rect)
				}
				yuvBufPool.Put(latestYUV.buf)
				yuvReady = true
				lastYUVFrameTime.Store(nowNs)
				fullScreenBitmapStartTime.Store(0)
				lastFullScreenBitmapTime.Store(0)
				renderDirty = 3
			}
		}

		// Drain ALL pending bitmap batches and combine into one paintImages call.
		// Processing every queued patch in a single call reduces cgo overhead,
		// eliminates catch-up lag on burst updates, and provides a larger batch
		// for the bounding-rect Lock optimisation when it is re-enabled.
		allBitmaps = allBitmaps[:0]
	drainBitmaps:
		for {
			select {
			case bs := <-bitmapCh:
				allBitmaps = append(allBitmaps, bs...)
			default:
				break drainBitmaps
			}
		}
		if len(allBitmaps) > 0 {
			prevLen := len(overlayDirtyRects)
			paintImages(allBitmaps, texture, width, height, &overlayDirtyRects)
			if len(overlayDirtyRects) > prevLen {
				overlayHasContent = true
			}
			// Track full-screen BGRA patches for the SW fallback watchdog.
			// A full-screen patch covering the whole window usually means the
			// server is sending video as BGRA because the H.264 hardware decoder
			// has stalled and grdp fell back to software decoding.
			hasFullScreen := false
			for _, bm := range allBitmaps {
				patchW := bm.DestRight - bm.DestLeft + 1
				patchH := bm.DestBottom - bm.DestTop + 1
				if patchW >= width && patchH >= height {
					hasFullScreen = true
					break
				}
			}
			if hasFullScreen {
				if fullScreenBitmapStartTime.Load() == 0 {
					fullScreenBitmapStartTime.Store(nowNs)
				}
				lastFullScreenBitmapTime.Store(nowNs)
			} else {
				fullScreenBitmapStartTime.Store(0)
				lastFullScreenBitmapTime.Store(0)
			}
			for i := range allBitmaps {
				bitmapBufPool.Put(allBitmaps[i].Data)
			}
			renderDirty = 3
		}

		// Render only when content has changed.  renderDirty counts down so that
		// every backbuffer is refreshed before we pause (SDL2 double/triple
		// buffering requires re-issuing the same draw commands until all buffers
		// are updated, otherwise a stale empty backbuffer flashes through).
		if renderDirty > 0 {
			// In H.264 sessions, keep the display black until the first video
			// frame arrives (or the warm-up grace elapses) so that intermediate
			// warm-up bitmap patches do not flicker on screen before the real
			// desktop baseline is ready.
			canShowContent := yuvReady || yuvTexture == nil || time.Duration(nowNs-neverShownConnectNs) >= neverShownKeyframeGrace

			if !yuvReady {
				renderer.SetDrawColor(0, 0, 0, 255)
				renderer.Clear()
			}
			if yuvReady {
				renderer.Copy(yuvTexture, nil, nil)
			}
			// Skip the overlay copy when the texture is fully transparent or when
			// content is held back during the initial connection warm-up.
			if canShowContent && (overlayHasContent || !yuvReady) {
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

		if resizePending && now.Sub(resizeTime) > 500*time.Millisecond {
			resizePending = false
			dw := abs(int(resizeW) - width)
			dh := abs(int(resizeH) - height)
			if dw > 2 || dh > 2 {
				slog.Info("Window resized, reconnecting", "width", resizeW, "height", resizeH, "oldWidth", width, "oldHeight", height)
				reconnecting.Store(1)
				reconnErr := rdpClient.Reconnect(int(resizeW), int(resizeH))
				reconnecting.Store(0)
				if reconnErr != nil {
					slog.Error("Reconnect failed", "err", reconnErr)
				} else {
					resetAfterReconnect(resizeW, resizeH)
				}
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

		if connectionErrorPending.CompareAndSwap(true, false) && !resizePending {
			w, h := window.GetSize()
			slog.Info("Connection error detected, reconnecting immediately", "width", w, "height", h)
			reconnecting.Store(1)
			reconnErr := rdpClient.Reconnect(int(w), int(h))
			reconnecting.Store(0)
			if reconnErr != nil {
				slog.Error("Reconnect (connection error) failed", "err", reconnErr)
				connectionErrorPending.Store(true)
			} else {
				resetAfterReconnect(w, h)
			}
		}

		// Never-shown-a-frame watchdog: reconnect once videoStallTimeout has
		// elapsed since the current connection attempt began, if we still
		// haven't shown a real frame.  This uses its own clock
		// (neverShownConnectNs) rather than lastServerActivity, because
		// lastServerActivity is refreshed by ANY server traffic — pointer
		// cursor updates in particular can arrive continuously (e.g. a
		// blinking/animated cursor) even while the video stream itself is
		// stuck on a dropped black warm-up frame, which would otherwise keep
		// resetting the stall clock and prevent this from ever firing.
		//
		// Before giving up and reconnecting, send a single ForceRefresh
		// keyframe request at neverShownKeyframeGrace: the server goes
		// completely silent after its initial black warm-up burst, and this
		// lightweight nudge can make it resend the now-painted desktop
		// without paying for a full reconnect.  Only one request is sent
		// (unlike the earlier 4-requests-every-2s ladder, which started
		// during the still-active warm-up burst and produced visibly
		// corrupted frames) — if it doesn't help, the reconnect below still
		// fires at videoStallTimeout.
		if !everShowedFrame.Load() && !resizePending {
			neverShownElapsed := time.Duration(nowNs - neverShownConnectNs)
			if neverShownElapsed >= neverShownKeyframeGrace &&
				(neverShownLastKeyframeNs == 0 || time.Duration(nowNs-neverShownLastKeyframeNs) >= 2*time.Second) {
				slog.Warn("Black screen, sending ForceRefresh and MouseMove",
					"sinceStart", neverShownElapsed.Round(time.Millisecond))
				rdpClient.MouseMove(500, 500)
				rdpClient.RequestKeyframe()
				neverShownLastKeyframeNs = nowNs
			}
			if neverShownElapsed > videoStallTimeout {
				w, h := window.GetSize()
				slog.Warn("Video stalled without ever showing a frame, reconnecting",
					"stalled", neverShownElapsed.Round(time.Millisecond), "width", w, "height", h)
				reconnecting.Store(1)
				reconnErr := rdpClient.Reconnect(int(w), int(h))
				reconnecting.Store(0)
				if reconnErr != nil {
					slog.Error("Reconnect (never shown) failed", "err", reconnErr)
					neverShownConnectNs = nowNs // retry next stall cycle
					neverShownLastKeyframeNs = 0
				} else {
					resetAfterReconnect(w, h)
				}
			}
		}

		// Video watchdog: reconnect if a connection error was reported and the
		// session has been silent for videoStallTimeout; otherwise just log
		// (the remote desktop may legitimately be idle for long periods).
		// The never-shown-a-frame case is handled by the watchdog above, so
		// this one only needs to cover a stall after a real frame has already
		// been shown.
		lastNS := lastServerActivity.Load()
		if lastNS != 0 && !resizePending {
			elapsed := time.Duration(nowNs - lastNS)
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
					lastServerActivity.Store(nowNs) // reset to avoid repeated log spam
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

	if int(scancode) < len(scancodeTable) {
		return scancodeTable[scancode]
	}
	return 0
}

// scancodeTable maps SDL2 scancode integers (0–511) to RDP scancode values.
// Direct array indexing avoids hash computation on every key event.
var scancodeTable [512]int

func init() {
	scancodeTable[sdl.SCANCODE_UNKNOWN] = 0x0000
	scancodeTable[sdl.SCANCODE_ESCAPE] = 0x0001
	scancodeTable[sdl.SCANCODE_1] = 0x0002
	scancodeTable[sdl.SCANCODE_2] = 0x0003
	scancodeTable[sdl.SCANCODE_3] = 0x0004
	scancodeTable[sdl.SCANCODE_4] = 0x0005
	scancodeTable[sdl.SCANCODE_5] = 0x0006
	scancodeTable[sdl.SCANCODE_6] = 0x0007
	scancodeTable[sdl.SCANCODE_7] = 0x0008
	scancodeTable[sdl.SCANCODE_8] = 0x0009
	scancodeTable[sdl.SCANCODE_9] = 0x000A
	scancodeTable[sdl.SCANCODE_0] = 0x000B
	scancodeTable[sdl.SCANCODE_MINUS] = 0x000C
	scancodeTable[sdl.SCANCODE_EQUALS] = 0x000D
	scancodeTable[sdl.SCANCODE_BACKSPACE] = 0x000E
	scancodeTable[sdl.SCANCODE_TAB] = 0x000F
	scancodeTable[sdl.SCANCODE_Q] = 0x0010
	scancodeTable[sdl.SCANCODE_W] = 0x0011
	scancodeTable[sdl.SCANCODE_E] = 0x0012
	scancodeTable[sdl.SCANCODE_R] = 0x0013
	scancodeTable[sdl.SCANCODE_T] = 0x0014
	scancodeTable[sdl.SCANCODE_Y] = 0x0015
	scancodeTable[sdl.SCANCODE_U] = 0x0016
	scancodeTable[sdl.SCANCODE_I] = 0x0017
	scancodeTable[sdl.SCANCODE_O] = 0x0018
	scancodeTable[sdl.SCANCODE_P] = 0x0019
	scancodeTable[sdl.SCANCODE_LEFTBRACKET] = 0x001A
	scancodeTable[sdl.SCANCODE_RIGHTBRACKET] = 0x001B
	scancodeTable[sdl.SCANCODE_RETURN] = 0x001C
	scancodeTable[sdl.SCANCODE_LCTRL] = 0x001D
	scancodeTable[sdl.SCANCODE_A] = 0x001E
	scancodeTable[sdl.SCANCODE_S] = 0x001F
	scancodeTable[sdl.SCANCODE_D] = 0x0020
	scancodeTable[sdl.SCANCODE_F] = 0x0021
	scancodeTable[sdl.SCANCODE_G] = 0x0022
	scancodeTable[sdl.SCANCODE_H] = 0x0023
	scancodeTable[sdl.SCANCODE_J] = 0x0024
	scancodeTable[sdl.SCANCODE_K] = 0x0025
	scancodeTable[sdl.SCANCODE_L] = 0x0026
	scancodeTable[sdl.SCANCODE_SEMICOLON] = 0x0027
	scancodeTable[sdl.SCANCODE_APOSTROPHE] = 0x0028
	scancodeTable[sdl.SCANCODE_GRAVE] = 0x0029
	scancodeTable[sdl.SCANCODE_LSHIFT] = 0x002A
	scancodeTable[sdl.SCANCODE_BACKSLASH] = 0x002B
	scancodeTable[sdl.SCANCODE_Z] = 0x002C
	scancodeTable[sdl.SCANCODE_X] = 0x002D
	scancodeTable[sdl.SCANCODE_C] = 0x002E
	scancodeTable[sdl.SCANCODE_V] = 0x002F
	scancodeTable[sdl.SCANCODE_B] = 0x0030
	scancodeTable[sdl.SCANCODE_N] = 0x0031
	scancodeTable[sdl.SCANCODE_M] = 0x0032
	scancodeTable[sdl.SCANCODE_COMMA] = 0x0033
	scancodeTable[sdl.SCANCODE_PERIOD] = 0x0034
	scancodeTable[sdl.SCANCODE_SLASH] = 0x0035
	scancodeTable[sdl.SCANCODE_RSHIFT] = 0x0036
	scancodeTable[sdl.SCANCODE_KP_MULTIPLY] = 0x0037
	scancodeTable[sdl.SCANCODE_LALT] = 0x0038
	scancodeTable[sdl.SCANCODE_SPACE] = 0x0039
	scancodeTable[sdl.SCANCODE_CAPSLOCK] = 0x003A
	scancodeTable[sdl.SCANCODE_F1] = 0x003B
	scancodeTable[sdl.SCANCODE_F2] = 0x003C
	scancodeTable[sdl.SCANCODE_F3] = 0x003D
	scancodeTable[sdl.SCANCODE_F4] = 0x003E
	scancodeTable[sdl.SCANCODE_F5] = 0x003F
	scancodeTable[sdl.SCANCODE_F6] = 0x0040
	scancodeTable[sdl.SCANCODE_F7] = 0x0041
	scancodeTable[sdl.SCANCODE_F8] = 0x0042
	scancodeTable[sdl.SCANCODE_F9] = 0x0043
	scancodeTable[sdl.SCANCODE_F10] = 0x0044
	scancodeTable[sdl.SCANCODE_SCROLLLOCK] = 0x0046
	scancodeTable[sdl.SCANCODE_KP_7] = 0x0047
	scancodeTable[sdl.SCANCODE_KP_8] = 0x0048
	scancodeTable[sdl.SCANCODE_KP_9] = 0x0049
	scancodeTable[sdl.SCANCODE_KP_MINUS] = 0x004A
	scancodeTable[sdl.SCANCODE_KP_4] = 0x004B
	scancodeTable[sdl.SCANCODE_KP_5] = 0x004C
	scancodeTable[sdl.SCANCODE_KP_6] = 0x004D
	scancodeTable[sdl.SCANCODE_KP_PLUS] = 0x004E
	scancodeTable[sdl.SCANCODE_KP_1] = 0x004F
	scancodeTable[sdl.SCANCODE_KP_2] = 0x0050
	scancodeTable[sdl.SCANCODE_KP_3] = 0x0051
	scancodeTable[sdl.SCANCODE_KP_0] = 0x0052
	scancodeTable[sdl.SCANCODE_KP_DECIMAL] = 0x0053
	scancodeTable[sdl.SCANCODE_F11] = 0x0057
	scancodeTable[sdl.SCANCODE_F12] = 0x0058
	scancodeTable[sdl.SCANCODE_KP_EQUALS] = 0x0059
	scancodeTable[sdl.SCANCODE_KP_ENTER] = 0xE01C
	scancodeTable[sdl.SCANCODE_RCTRL] = 0xE01D
	scancodeTable[sdl.SCANCODE_KP_DIVIDE] = 0xE035
	scancodeTable[sdl.SCANCODE_PRINTSCREEN] = 0xE037
	scancodeTable[sdl.SCANCODE_RALT] = 0xE038
	scancodeTable[sdl.SCANCODE_NUMLOCKCLEAR] = 0xE045
	scancodeTable[sdl.SCANCODE_PAUSE] = 0xE046
	scancodeTable[sdl.SCANCODE_HOME] = 0xE047
	scancodeTable[sdl.SCANCODE_UP] = 0xE048
	scancodeTable[sdl.SCANCODE_PAGEUP] = 0xE049
	scancodeTable[sdl.SCANCODE_LEFT] = 0xE04B
	scancodeTable[sdl.SCANCODE_RIGHT] = 0xE04D
	scancodeTable[sdl.SCANCODE_END] = 0xE04F
	scancodeTable[sdl.SCANCODE_DOWN] = 0xE050
	scancodeTable[sdl.SCANCODE_PAGEDOWN] = 0xE051
	scancodeTable[sdl.SCANCODE_INSERT] = 0xE052
	scancodeTable[sdl.SCANCODE_DELETE] = 0xE053
	scancodeTable[sdl.SCANCODE_MENU] = 0xE05D
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
	disableAVC444 := flag.Bool("disable-avc444", false, "disable AVC444/AVC444v2 and use AVC420 only")
	flag.Parse()

	if *debugLog {
		handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})
		slog.SetDefault(slog.New(handler))
	}
	slog.Debug("flag", "swap_alt_meta", *swapAltMeta, "debug", *debugLog, "disable_avc444", *disableAVC444)

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

	mainLoop(hostPort, domain, user, password, width, height, *swapAltMeta, keyboardType, keyboardLayout, *disableAVC444)
}
