# LiveCast

WebRTC SFU demo (rooms, captions, face AI) for [Render](https://render.com).

## Deploy on Render

1. Push this repo to GitHub.
2. Render Dashboard → **New** → **Blueprint** → select this repo (uses `render.yaml`).
3. After deploy, open `https://<service>.onrender.com`.
4. **Required for cloud WebRTC:** add free Metered TURN credentials (Render has no public UDP — without TURN, Host preview works but media never reaches the SFU, so Join stays on “waiting”).

### Metered TURN (free, ~5–20 GB/mo)

1. Sign up: https://www.metered.ca/tools/openrelay/
2. Dashboard → create a TURN credential / copy **API Key**.
3. Note your app domain, e.g. `yourappname.metered.live`.
4. On Render → your service → **Environment**, add:

| Key | Value |
| --- | --- |
| `METERED_DOMAIN` | `yourappname.metered.live` |
| `METERED_API_KEY` | *(API key from Metered)* |

Alternative (static username/password from the same dashboard):

| Key | Example |
| --- | --- |
| `TURN_URLS` | `turn:global.relay.metered.ca:80,turn:global.relay.metered.ca:80?transport=tcp,turns:global.relay.metered.ca:443?transport=tcp` |
| `TURN_USERNAME` | *(from Metered)* |
| `TURN_CREDENTIAL` | *(from Metered)* |

5. Redeploy, then open `/status` — you want `"turn_ready": true`.

Render terminates HTTPS; the app listens on `$PORT` over HTTP.

**Note:** Same-LAN demos work better with a local build (`go run . -listen 0.0.0.0 -https`) — no Metered needed.

## Reliability notes

- Idle rooms without media are GC’d after ~3 minutes.
- Host **End** closes the peer connection (room code stays for a retry until idle GC).
- Cloud Go live is blocked until `/status` reports `turn_ready: true`.

## Local

```bash
go run .
# or
docker build -t livecast . && docker run --rm -p 10000:10000 -e PORT=10000 livecast
```
