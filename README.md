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

| Variable               | Description                          | Default    |
|------------------------|--------------------------------------|------------|
| `GRDP_HOST`            | RDP server hostname or IP address    | (required) |
| `GRDP_PORT`            | RDP server port                      | (required) |
| `GRDP_USER`            | Username                             | (required) |
| `GRDP_PASSWORD`        | Password                             | (required) |
| `GRDP_DOMAIN`          | Domain (if required by your account) | (empty)    |
| `GRDP_WINDOW_SIZE`     | Window size in `WxH` format          | `1280x800` |
| `GRDP_KEYBOARD_TYPE`   | Keyboard type (see below)            | `IBM_101_102_KEYS` |
| `GRDP_KEYBOARD_LAYOUT` | Keyboard layout (see below)          | `US`       |

Example:

```sh
export GRDP_HOST=myserver
export GRDP_PORT=3389
export GRDP_USER=user
export GRDP_PASSWORD=password
export GRDP_WINDOW_SIZE=1280x800
export GRDP_KEYBOARD_TYPE=IBM_101_102_KEYS
export GRDP_KEYBOARD_LAYOUT=JAPANESE
```

### Keyboard Types (`GRDP_KEYBOARD_TYPE`)

| Value              | Description              |
|--------------------|--------------------------|
| `IBM_PC_XT_83_KEY` | IBM PC/XT 83-key         |
| `OLIVETTI`         | Olivetti                 |
| `IBM_PC_AT_84_KEY` | IBM PC/AT 84-key         |
| `IBM_101_102_KEYS` | IBM 101/102-key (default)|
| `NOKIA_1050`       | Nokia 1050               |
| `NOKIA_9140`       | Nokia 9140               |
| `JAPANESE`         | Japanese                 |

### Keyboard Layouts (`GRDP_KEYBOARD_LAYOUT`)

| Value                 | Description              |
|-----------------------|--------------------------|
| `ARABIC`              | Arabic                   |
| `BULGARIAN`           | Bulgarian                |
| `CHINESE_US_KEYBOARD` | Chinese (US keyboard)    |
| `CZECH`               | Czech                    |
| `DANISH`              | Danish                   |
| `GERMAN`              | German                   |
| `GREEK`               | Greek                    |
| `US`                  | US English (default)     |
| `SPANISH`             | Spanish                  |
| `FINNISH`             | Finnish                  |
| `FRENCH`              | French                   |
| `HEBREW`              | Hebrew                   |
| `HUNGARIAN`           | Hungarian                |
| `ICELANDIC`           | Icelandic                |
| `ITALIAN`             | Italian                  |
| `JAPANESE`            | Japanese                 |
| `KOREAN`              | Korean                   |
| `DUTCH`               | Dutch                    |
| `NORWEGIAN`           | Norwegian                |

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
