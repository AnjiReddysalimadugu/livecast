# LiveCast

Complete WebRTC SFU app (rooms, captions, translation, face AI).

**Develop locally from:** `webrtc/examples/broadcast`  
This GitHub repo is the deployable copy for Render.

## Render

- Dockerfile at repo root
- Env: `METERED_DOMAIN`, `METERED_API_KEY` (Metered Open Relay)
- `/status` → `"turn_ready": true`

## Local

```bash
go run .
# or
go build -o livecast.exe .
./livecast.exe -port 8081 -listen 0.0.0.0 -https
```
