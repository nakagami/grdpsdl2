# grdpsdl2

Microsoft RDP client built with [grdp](https://github.com/nakagami/grdp) and [go-sdl2](https://github.com/veandco/go-sdl2).

## Installation

```sh
go install github.com/nakagami/grdpsdl2@latest
```

## Requirements

Install SDL2 by following the instructions [here](https://github.com/veandco/go-sdl2?tab=readme-ov-file#requirements).

## Configuration

Connection parameters are specified via environment variables.

| Variable           | Description                          | Default    |
|--------------------|--------------------------------------|------------|
| `GRDP_HOST`        | RDP server hostname or IP address    | (required) |
| `GRDP_PORT`        | RDP server port                      | (required) |
| `GRDP_USER`        | Username                             | (required) |
| `GRDP_PASSWORD`    | Password                             | (required) |
| `GRDP_DOMAIN`      | Domain (if required by your account) | (empty)    |
| `GRDP_WINDOW_SIZE` | Window size in `WxH` format          | `1280x800` |

Example:

```sh
export GRDP_HOST=myserver
export GRDP_PORT=3389
export GRDP_USER=user
export GRDP_PASSWORD=password
export GRDP_WINDOW_SIZE=1280x800
```

## Usage

```sh
grdpsdl2 [options]
```

### Options

#### `-swap-alt-meta`

Swaps the Alt key and the Meta (GUI/Super/Windows) key.

This is useful on macOS, where the Command key (⌘) is reported as the Meta/GUI key.
When connecting to a Windows RDP session, you may want the Command key to behave as
the Alt key (and vice versa). Passing `-swap-alt-meta` enables this mapping.

```sh
grdpsdl2 -swap-alt-meta
```
