# LiveCast (examples/broadcast)

WebRTC SFU demo: rooms, captions, translation, face AI, browser voice replies.  
**This folder is the source of truth** — use `C:\Users\anjir\Desktop\webrtc\examples\broadcast`.

**Phone / SIP AI (option 3 dual stack):** see sibling project  
`C:\Users\anjir\Desktop\livekit-phone` (LiveKit + Sinch EST on a VPS).

## Local (LAN)

```bat
cd examples\broadcast
build.bat
broadcast.exe -port 8081 -listen 0.0.0.0 -https
```

Open `https://<PC-LAN-IP>:8081` (accept cert warning).

## Cloud (Render)

Deploy the **flat LiveCast app** (recommended): GitHub `AnjiReddysalimadugu/livecast`  
(Dockerfile expects `go.mod` + `web/` at repo root — that is the livecast mirror).

Do **not** point Render `dockerContext` at the full `pion/webrtc` monorepo root.

Set env on Render:

| Key | Value |
| --- | --- |
| `METERED_DOMAIN` | `yourapp.metered.live` |
| `METERED_API_KEY` | *(from Metered dashboard)* |
| `OPENAI_API_KEY` | *(optional — better STT + AI replies)* |

After deploy, `/status` should show `"turn_ready": true`.

## Why not a separate `livecast` folder?

A standalone `Desktop\livecast` copy was used only as a thin Render deploy mirror (own `go.mod`).  
Edit `examples/broadcast`, then sync/push the livecast mirror for Render.
