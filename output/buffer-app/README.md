# Buffered URL Text Viewer

A small web app that streams text from a URL and throttles display speed.

## Features
- Single URL input
- Throttle by chars/sec (0 = no throttle)
- Live output with optional auto-scroll
- Click output area to toggle auto-scroll
- Hotkey: Ctrl+Shift+S (Cmd+Shift+S on macOS)
- Max total time and stall timeout controls
- Download buffered text to a local file
- Local proxy to avoid CORS issues

## Run
1. Start the server:
   - node server.js
2. Open:
   - http://localhost:3000

## Install globally
1. Run:
   - sudo ./install.sh
2. Start:
   - buffer-app
3. Open:
   - http://localhost:3000

Install path:
- /usr/local/share/buffer-app
- /usr/local/bin/buffer-app

## Notes
- The proxy endpoint is `/proxy?url=...` and accepts http/https URLs.
