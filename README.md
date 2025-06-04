# grdpsdl2
Microsoft RDP client.

It depends on [grdp](https://github.com/nakagami/grdp) and [go-sdl2](https://github.com/veandco/go-sdl2).

See [here](https://github.com/veandco/go-sdl2?tab=readme-ov-file#requirements) and install SDL2.

# How to use

Currently, the required arguments are taken from environment variables.

For example
```
export GRDP_USER=user
export GRDP_PASSWORD=password
export GRDP_PORT=3389
export GRDP_HOST=host
export GRDP_WINDOW_SIZE=1280x800
```
In your environment. You may also need to set `GRDP_DOMAIN`
