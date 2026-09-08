# LiveCast

WebRTC SFU demo (rooms, captions, face AI) for [Render](https://render.com).

## Deploy on Render

1. Push this repo to GitHub.
2. Render Dashboard → **New** → **Blueprint** → select this repo (uses `render.yaml`).
3. After deploy, open `https://<service>.onrender.com`.
4. (Recommended) Add free TURN credentials as env vars so people on other networks can join:
   - `TURN_URLS` e.g. `turn:openrelay.metered.ca:80,turn:openrelay.metered.ca:443`
   - `TURN_USERNAME` / `TURN_CREDENTIAL` from [Metered](https://www.metered.ca/tools/openrelay/)

Render terminates HTTPS; the app listens on `$PORT` over HTTP.

**Note:** Render does not expose UDP. Cross-network WebRTC often needs TURN. Same LAN demos work better with the local `broadcast.exe -listen 0.0.0.0 -https` flow.

## Local

```bash
go run .
# or
docker build -t livecast . && docker run --rm -p 10000:10000 -e PORT=10000 livecast
```
