package main

import (
	"image"
	"log/slog"
	"os"
	"strings"
	"unsafe"

	"github.com/nakagami/grdp"
	"github.com/veandco/go-sdl2/sdl"
)

func paintImage(img *image.RGBA, surface *sdl.Surface, destX, destY int) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	surfW, surfH := int(surface.W), int(surface.H)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sx, sy := destX+x, destY+y
			if sx < 0 || sy < 0 || sx >= surfW || sy >= surfH {
				continue
			}
			offset := img.PixOffset(x, y)
			r := img.Pix[offset+0]
			g := img.Pix[offset+1]
			b := img.Pix[offset+2]
			a := img.Pix[offset+3]
			color := sdl.MapRGBA(surface.Format, r, g, b, a)

			pixels := surface.Pixels()
			ptr := uintptr(unsafe.Pointer(&pixels[0]))
			pitch := int(surface.Pitch)
			pxOffset := sy*pitch + sx*4 // RGBA32bit
			*(*uint32)(unsafe.Pointer(ptr + uintptr(pxOffset))) = color
		}
	}
}

func mainLoop(hostPort, domain, user, password string, width, height int) (err error) {
	cursorCache := make(map[uint16]*sdl.Cursor)
	showCursor := true

	if err = sdl.Init(sdl.INIT_VIDEO); err != nil {
		return err
	}
	defer sdl.Quit()

	sdl.StopTextInput()

	window, err := sdl.CreateWindow("GRDPSDL2", sdl.WINDOWPOS_UNDEFINED,
		sdl.WINDOWPOS_UNDEFINED, int32(width), int32(height), sdl.WINDOW_SHOWN)
	if err != nil {
		return err
	}
	surface, _ := window.GetSurface()

	rdpClient := grdp.NewRdpClient(hostPort, width, height)
	err = rdpClient.Login(domain, user, password)
	if err != nil {
		return err
	}

	rdpClient.OnError(func(e error) {
		slog.Error("on error", "err", e)
	}).OnReady(func() {
		slog.Info("on ready")
	}).OnBitmap(func(bs []grdp.Bitmap) {
		surface.Lock()
		defer surface.Unlock()
		for _, bm := range bs {
			paintImage(bm.RGBA(), surface, bm.DestLeft, bm.DestTop)
		}
		window.UpdateSurface()
	}).OnPointerHide(func() {
		sdl.ShowCursor(sdl.DISABLE)
		showCursor = false
	}).OnPointerCached(func(idx uint16) {
		if !showCursor {
			sdl.ShowCursor(sdl.ENABLE)
			showCursor = true
		}
		sdl.SetCursor(cursorCache[idx])
	}).OnPointerUpdate(func(idx uint16, bpp uint16, x uint16, y uint16, width uint16, height uint16, mask []byte, data []byte) {
		if !showCursor {
			sdl.ShowCursor(sdl.ENABLE)
			showCursor = true
		}
		if bpp != 32 {
			slog.Error("Can't update Pointer", "bpp", bpp)
			return
		}

		// I don't know why, but there is a strange bitmap on the bottom line.
		height -= 1

		surface, err := sdl.CreateRGBSurfaceWithFormatFrom(
			unsafe.Pointer(&data[0]), int32(width), int32(height), 32, int32(width*4), uint32(sdl.PIXELFORMAT_RGBA32))
		if err != nil {
			slog.Error("surface", "err", err, "bpp", bpp, "width", width, "height", height, "len(data)", len(data))
		}

		// swap lines
		line_len := int(width) * 4
		upper_line := 0
		lower_line := int(height) - 1
		for upper_line < int(height)/2 {
			for i := 0; i < line_len; i++ {
				data[upper_line*line_len+i], data[lower_line*line_len+i] = data[lower_line*line_len+i], data[upper_line*line_len+i]
			}
			upper_line += 1
			lower_line -= 1
		}

		cursor := sdl.CreateColorCursor(surface, int32(x), int32(y))
		cursorCache[idx] = cursor
		if cursor == nil {
			slog.Error("Failed to create cursor")
		}
		sdl.SetCursor(cursor)

	})

	quit := false
	for !quit {
		for event := sdl.PollEvent(); event != nil; event = sdl.PollEvent() {
			switch t := event.(type) {
			case *sdl.QuitEvent:
				quit = true

			case *sdl.KeyboardEvent:
				if t.State == sdl.RELEASED {
					rdpClient.KeyUp(transKey(t.Keysym.Scancode))
				} else if t.State == sdl.PRESSED {
					rdpClient.KeyDown(transKey(t.Keysym.Scancode))
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
					rdpClient.MouseWheel(int(t.Y))
				}
			}
		}
	}

	err = window.Destroy()
	return err
}

func transKey(scancode sdl.Scancode) int {

	var ScancodeMap = map[sdl.Scancode]int{
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
	if v, ok := ScancodeMap[scancode]; ok {
		return v
	}
	return 0
}

func main() {
	//    handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})
	//    slog.SetDefault(slog.New(handler))
	hostPort := strings.Join([]string{os.Getenv("GRDP_HOST"), os.Getenv("GRDP_PORT")}, ":")
	domain := os.Getenv("GRDP_DOMAIN")
	user := os.Getenv("GRDP_USER")
	password := os.Getenv("GRDP_PASSWORD")
	width := 1280
	height := 800

	mainLoop(hostPort, domain, user, password, width, height)
}
