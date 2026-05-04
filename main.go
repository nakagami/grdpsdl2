package main

import (
	"flag"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/nakagami/grdp"
	"github.com/nakagami/grdp/plugin/rdpsnd"
	"github.com/veandco/go-sdl2/sdl"
)

func paintImages(bs []grdp.Bitmap, texture *sdl.Texture) {
	for _, bm := range bs {
		// The texture uses PIXELFORMAT_BGRA32, so grdp's native BGRA data
		// (BitsPerPixel==4) can be passed directly — no copy or byte-swap needed.
		// For legacy bit-depths (2/3 bpp), bm.RGBA() converts to RGBA; we then
		// swap R↔B in-place to match the BGRA32 texture format.  Those paths
		// are uncommon outside of traditional RDP bitmap updates.
		if bm.BitsPerPixel == 4 {
			// Fast path: BGRA data passes straight to the BGRA32 texture.
			w := bm.DestRight - bm.DestLeft + 1
			if w > bm.Width {
				w = bm.Width
			}
			h := bm.DestBottom - bm.DestTop + 1
			if h > bm.Height {
				h = bm.Height
			}
			rect := sdl.Rect{X: int32(bm.DestLeft), Y: int32(bm.DestTop), W: int32(w), H: int32(h)}
			texture.Update(&rect, unsafe.Pointer(&bm.Data[0]), bm.Width*4)
		} else {
			// Slow path: convert to RGBA, then swap R↔B for BGRA32 texture.
			// Use the smaller of the destination rectangle and the actual
			// image dimensions (same clamping as before).
			img := bm.RGBA()
			w := bm.DestRight - bm.DestLeft + 1
			if imgW := img.Bounds().Dx(); w > imgW {
				w = imgW
			}
			h := bm.DestBottom - bm.DestTop + 1
			if imgH := img.Bounds().Dy(); h > imgH {
				h = imgH
			}
			p := img.Pix
			for i := 0; i < len(p); i += 4 {
				p[i], p[i+2] = p[i+2], p[i]
			}
			rect := sdl.Rect{X: int32(bm.DestLeft), Y: int32(bm.DestTop), W: int32(w), H: int32(h)}
			texture.Update(&rect, unsafe.Pointer(&img.Pix[0]), img.Stride)
		}
	}
}

// audioPlayer manages SDL2 audio device for RDPSND playback.
// The device is opened once on the main thread at startup with a fixed format
// (44100 Hz / stereo / S16LE). play() only calls sdl.QueueAudio which is
// thread-safe and can be invoked from any goroutine.
type audioPlayer struct {
	deviceID sdl.AudioDeviceID
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

func (a *audioPlayer) play(af rdpsnd.AudioFormat, data []byte) {
	if a.deviceID == 0 {
		return
	}
	if err := sdl.QueueAudio(a.deviceID, data); err != nil {
		slog.Error("audio: QueueAudio", "err", err)
	}
}

func (a *audioPlayer) close() {
	if a.deviceID != 0 {
		sdl.CloseAudioDevice(a.deviceID)
		a.deviceID = 0
	}
}

// yuvFrame carries a decoded H.264 frame in I420 planar format from the grdp
// callback to the SDL2 main thread.  buf is the single backing allocation that
// holds all three planes; it is returned to yuvBufPool after the texture upload.
type yuvFrame struct {
	destX, destY, w, h int
	y                  []byte
	yStride            int
	u                  []byte
	uStride            int
	v                  []byte
	vStride            int
	buf                []byte // entire backing store: y||u||v
}

func mainLoop(hostPort, domain, user, password string, width, height int, swap_alt_meta bool, keyboardType, keyboardLayout string) (err error) {
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
	renderer, err := sdl.CreateRenderer(window, -1, sdl.RENDERER_ACCELERATED)
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

	// yuvTexture holds the most recent H264 frame in I420 (IYUV) format.
	// SDL2's renderer uses hardware YUV→RGB shaders for IYUV textures,
	// offloading colour conversion entirely from the CPU.
	// On software renderers SDL2 does the conversion in software — no separate
	// GPU/non-GPU code path is needed.
	yuvTexture, err := renderer.CreateTexture(uint32(sdl.PIXELFORMAT_IYUV), sdl.TEXTUREACCESS_STREAMING, int32(width), int32(height))
	if err != nil {
		// IYUV unsupported (unusual but possible on some drivers); fall back to
		// BGRA-only rendering by setting yuvTexture to nil.
		slog.Warn("IYUV texture unsupported, H264 will render via BGRA fallback", "err", err)
		yuvTexture = nil
	}
	if yuvTexture != nil {
		defer yuvTexture.Destroy()
	}

	// overlayZero is a pre-zeroed buffer used to reset the overlay texture to
	// fully transparent (BGRA 0,0,0,0) after each H264 full-frame update,
	// ensuring stale non-H264 patches do not obscure the new H264 baseline.
	// Allocated once; reused on every H264 frame and on texture recreation.
	overlayZero := make([]byte, width*height*4)
	// Initialise texture to transparent now so blending is correct from the first frame.
	texture.Update(nil, unsafe.Pointer(&overlayZero[0]), width*4)

	bitmapCh := make(chan []grdp.Bitmap, 128)
	yuvCh := make(chan yuvFrame, 128)
	yuvReady := false // true once any H264 I420 frame has been rendered
	clipboardFromServer := make(chan string, 4)
	clipboardReqCh := make(chan chan string, 1)

	// lastServerActivity tracks the last time we received any data from the
	// server (bitmap, pointer update, audio, clipboard).  Used by the video
	// stall watchdog below to distinguish a truly stuck stream from an idle
	// remote desktop.  Stored as UnixNano via atomic so it can be updated
	// from network goroutines and read from the main loop without a mutex.
	// Zero means no server activity yet (watchdog disarmed).
	var lastServerActivity atomic.Int64

	// Register a custom SDL event type to wake the main loop when bitmaps arrive.
	bitmapEventType := sdl.RegisterEvents(1)

	rdpClient := grdp.NewRdpClient(hostPort, width, height, func(hostPort string) (net.Conn, error) {
		return net.Dial("tcp", hostPort)
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
	}).OnReady(func() {
		slog.Info("on ready")
	}).OnAudio(func(af rdpsnd.AudioFormat, data []byte) {
		lastServerActivity.Store(time.Now().UnixNano())
		ap.play(af, data)
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
		if sent && bitmapEventType != sdl.FIRSTEVENT {
			sdl.PushEvent(&sdl.UserEvent{Type: bitmapEventType})
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
			for i := 0; i < n; i++ {
				b, g, r := data[3*i], data[3*i+1], data[3*i+2]
				// Branchless alpha: 0x00 when all channels are zero, 0xFF otherwise.
				// This avoids a conditional branch that would inhibit auto-vectorisation.
				var a byte
				if b|g|r != 0 {
					a = 0xFF
				}
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
	})

	// OnH264I420 receives decoded H264 frames in I420 planar format.
	// When the SDL2 IYUV texture is available this callback is called instead
	// of OnBitmap for H264 frames, so the YUV→RGB conversion is done by the
	// GPU (or SDL2's software renderer) rather than on the CPU.
	// The plane slices are valid only for the duration of the callback; we
	// copy them into a pooled buffer and send to yuvCh for the main thread.
	if yuvTexture != nil {
		rdpClient.OnH264I420(func(destX, destY, w, h int, y []byte, yStride int, u []byte, uStride int, v []byte, vStride int) {
			lastServerActivity.Store(time.Now().UnixNano())
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
				y: buf[:yLen], yStride: yStride,
				u: buf[yLen : yLen+uLen], uStride: uStride,
				v: buf[yLen+uLen : yLen+uLen+vLen], vStride: vStride,
				buf: buf,
			}
			sent := false
			select {
			case yuvCh <- frame:
				sent = true
			default:
				yuvBufPool.Put(buf)
				slog.Warn("yuv channel full, dropping H264 frame")
			}
			if sent && bitmapEventType != sdl.FIRSTEVENT {
				sdl.PushEvent(&sdl.UserEvent{Type: bitmapEventType})
			}
		})
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

	quit := false
	var resizePending bool
	var resizeTime time.Time
	var resizeW, resizeH int32
	var lastClipboardText string
	lastClipboardCheck := time.Now()

	for !quit {
		event := sdl.WaitEventTimeout(8)
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
				k := transKey(t.Keysym.Scancode, swap_alt_meta)
				if t.State == sdl.RELEASED {
					rdpClient.KeyUp(k)
				} else if t.State == sdl.PRESSED {
					rdpClient.KeyDown(k)
				}

			case *sdl.MouseMotionEvent:
				rdpClient.MouseMove(int(t.X), int(t.Y))

			case *sdl.MouseButtonEvent:
				if t.State == sdl.PRESSED {
					rdpClient.MouseDown(int(t.Button)-1, int(t.X), int(t.Y))
				} else {
					rdpClient.MouseUp(int(t.Button-1), int(t.X), int(t.Y))
				}

			case *sdl.MouseWheelEvent:
				if t.X == 0 {
					rdpClient.MouseWheel(int(t.Y) * 10)
				}
			}
		}

		// Drain incoming bitmaps and update GPU texture on the main thread.
		dirty := false

		// Drain H264 I420 frames: upload to IYUV texture and clear the overlay
		// so stale non-H264 patches don't obscure the new baseline frame.
	drainYUV:
		for {
			select {
			case frame := <-yuvCh:
				// Clear overlay to transparent: the new I420 frame is the fresh baseline.
				texture.Update(nil, unsafe.Pointer(&overlayZero[0]), width*4)
				rect := sdl.Rect{X: int32(frame.destX), Y: int32(frame.destY), W: int32(frame.w), H: int32(frame.h)}
				yuvTexture.UpdateYUV(&rect, frame.y, frame.yStride, frame.u, frame.uStride, frame.v, frame.vStride)
				yuvBufPool.Put(frame.buf)
				yuvReady = true
				dirty = true
			default:
				break drainYUV
			}
		}

	drain:
		for {
			select {
			case bs := <-bitmapCh:
				paintImages(bs, texture)
				for i := range bs {
					bitmapBufPool.Put(bs[i].Data)
				}
				dirty = true
			default:
				break drain
			}
		}
		if dirty {
			if yuvReady {
				// H264 session: render IYUV base first, then overlay patches on top.
				renderer.Copy(yuvTexture, nil, nil)
			}
			// Always render the overlay (non-H264 bitmaps, or full content when yuvReady=false).
			renderer.Copy(texture, nil, nil)
			renderer.Present()
		}

		// Handle clipboard from server (server → client)
		select {
		case text := <-clipboardFromServer:
			sdl.SetClipboardText(text)
			// Don't update lastClipboardText here so that the next poll
			// detects the change and calls NotifyClipboardChanged(),
			// which consumes grdp's suppressNextLocalChange flag.
		default:
		}

		// Handle clipboard request from server (client → server)
		select {
		case respCh := <-clipboardReqCh:
			text, _ := sdl.GetClipboardText()
			respCh <- text
		default:
		}

		// Poll local clipboard changes
		if time.Since(lastClipboardCheck) > 500*time.Millisecond {
			lastClipboardCheck = time.Now()
			if text, err := sdl.GetClipboardText(); err == nil && text != lastClipboardText {
				lastClipboardText = text
				rdpClient.NotifyClipboardChanged()
			}
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
				lastServerActivity.Store(0) // reset watchdog for fresh session
				texture.Destroy()
				texture, err = renderer.CreateTexture(uint32(sdl.PIXELFORMAT_BGRA32), sdl.TEXTUREACCESS_STREAMING, resizeW, resizeH)
				if err != nil {
					slog.Error("CreateTexture after resize failed", "err", err)
				} else {
					texture.SetBlendMode(sdl.BLENDMODE_BLEND)
					overlayZero = make([]byte, int(resizeW)*int(resizeH)*4)
					texture.Update(nil, unsafe.Pointer(&overlayZero[0]), int(resizeW)*4)
				}
				if yuvTexture != nil {
					yuvTexture.Destroy()
					yuvTexture, err = renderer.CreateTexture(uint32(sdl.PIXELFORMAT_IYUV), sdl.TEXTUREACCESS_STREAMING, resizeW, resizeH)
					if err != nil {
						slog.Warn("IYUV recreate failed after resize", "err", err)
						yuvTexture = nil
					}
				}
				yuvReady = false
			}
		}

		// Video watchdog: if no traffic of any kind arrives from the server
		// for a long time after the session was active, the H.264 decoder
		// may be stuck (e.g. after an HW→SW fallback where the server never
		// resends an IDR).  Reconnect to reset the decoder and recover the
		// stream.  We track all server-originated activity (bitmaps,
		// pointer updates, audio, clipboard) so that an idle desktop with
		// nothing to redraw does not trigger a false-positive reconnect.
		lastNS := lastServerActivity.Load()
		if lastNS != 0 && !resizePending {
			elapsed := time.Since(time.Unix(0, lastNS))
			if elapsed > videoStallTimeout {
				slog.Warn("Video stalled, reconnecting to recover",
					"stalled", elapsed.Round(time.Millisecond))
				lastServerActivity.Store(time.Now().UnixNano()) // prevent repeated reconnects while Reconnect runs
				curW, curH := window.GetSize()
				reconnecting.Store(1)
				reconnErr := rdpClient.Reconnect(int(curW), int(curH))
				reconnecting.Store(0)
				if reconnErr != nil {
					slog.Error("Video stall reconnect failed", "err", reconnErr)
				} else {
					lastServerActivity.Store(0) // reset watchdog for fresh session
					texture.Destroy()
					texture, err = renderer.CreateTexture(uint32(sdl.PIXELFORMAT_BGRA32), sdl.TEXTUREACCESS_STREAMING, curW, curH)
					if err != nil {
						slog.Error("CreateTexture after stall reconnect failed", "err", err)
					} else {
						texture.SetBlendMode(sdl.BLENDMODE_BLEND)
						overlayZero = make([]byte, int(curW)*int(curH)*4)
						texture.Update(nil, unsafe.Pointer(&overlayZero[0]), int(curW)*4)
					}
					if yuvTexture != nil {
						yuvTexture.Destroy()
						yuvTexture, err = renderer.CreateTexture(uint32(sdl.PIXELFORMAT_IYUV), sdl.TEXTUREACCESS_STREAMING, curW, curH)
						if err != nil {
							slog.Warn("IYUV recreate failed after stall reconnect", "err", err)
							yuvTexture = nil
						}
					}
					yuvReady = false
				}
			}
		}
	}

	err = window.Destroy()
	return err
}

func transKey(scancode sdl.Scancode, trans_alt_meta bool) int {
	if trans_alt_meta {
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
	// handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})
	// slog.SetDefault(slog.New(handler))

	swap_alt_meta := flag.Bool("swap-alt-meta", false, "swap alt and meta key")
	flag.Parse()
	slog.Debug("flag", "swap_alt_meta", *swap_alt_meta)

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

	mainLoop(hostPort, domain, user, password, width, height, *swap_alt_meta, keyboardType, keyboardLayout)
}
