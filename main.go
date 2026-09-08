// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

//go:build !js

// broadcast demonstrates SFU audio/video broadcast with server-side face detection.
package main

import (
	"bufio"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/pion/interceptor"
	"github.com/pion/interceptor/pkg/intervalpli"
	"github.com/pion/rtp/codecs"
	"github.com/pion/webrtc/v4"
	"github.com/pion/webrtc/v4/pkg/media/samplebuilder"
	"golang.org/x/net/websocket"
)

type faceHub struct {
	mu     sync.Mutex
	last   []byte
	chans  []*webrtc.DataChannel
	events *faceEventLogger
}

func (h *faceHub) snapshot() []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.last) == 0 {
		return []byte(`{"present":false,"count":0,"status":"waiting for frames","confidence":0}`)
	}
	return append([]byte(nil), h.last...)
}

func (h *faceHub) reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.chans = nil
	// keep last face status so /face polling still works across reconnects
}

func (h *faceHub) add(dc *webrtc.DataChannel) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.chans = append(h.chans, dc)

	dc.OnOpen(func() {
		h.mu.Lock()
		last := append([]byte(nil), h.last...)
		h.mu.Unlock()
		if len(last) == 0 {
			return
		}
		_ = dc.SendText(string(last))
	})
}

func (h *faceHub) broadcast(msg []byte) {
	h.mu.Lock()
	h.last = append([]byte(nil), msg...)
	chans := append([]*webrtc.DataChannel(nil), h.chans...)
	events := h.events
	h.mu.Unlock()

	if events != nil {
		events.logChange(msg)
	}
	for _, dc := range chans {
		if dc == nil || dc.ReadyState() != webrtc.DataChannelStateOpen {
			continue
		}
		_ = dc.SendText(string(msg))
	}
}

type faceWorker struct {
	stdin  io.WriteCloser
	cmd    *exec.Cmd
	mu     sync.Mutex
	frames chan []byte
	once   sync.Once
}

func startFaceWorker(hub *faceHub) (*faceWorker, error) {
	script, err := filepath.Abs(filepath.Join("facedetect", "worker.py"))
	if err != nil {
		return nil, err
	}
	cmd := exec.Command("python", "-u", script)
	cmd.Stderr = os.Stderr
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	w := &faceWorker{stdin: stdin, cmd: cmd, frames: make(chan []byte, 1)}

	go func() {
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for sc.Scan() {
			line := append([]byte(nil), sc.Bytes()...)
			hub.broadcast(line)
			fmt.Printf("face-detect: %s\n", line)
		}
	}()

	go func() {
		for frame := range w.frames {
			w.mu.Lock()
			var hdr [4]byte
			binary.BigEndian.PutUint32(hdr[:], uint32(len(frame)))
			_, err := w.stdin.Write(hdr[:])
			if err == nil {
				_, err = w.stdin.Write(frame)
			}
			w.mu.Unlock()
			if err != nil {
				fmt.Printf("face-worker write error: %v\n", err)
				return
			}
		}
	}()

	fmt.Println("Server-side face detection worker started (YuNet local model)")
	return w, nil
}

func (w *faceWorker) submit(frame []byte) {
	if w == nil {
		return
	}
	select {
	case w.frames <- frame:
	default:
		select {
		case <-w.frames:
		default:
		}
		select {
		case w.frames <- frame:
		default:
		}
	}
}

func (w *faceWorker) Close() {
	if w == nil {
		return
	}
	w.once.Do(func() {
		close(w.frames)
		_ = w.stdin.Close()
		if w.cmd != nil && w.cmd.Process != nil {
			_ = w.cmd.Process.Kill()
		}
	})
}

func createFacesChannel(pc *webrtc.PeerConnection, hub *faceHub) error {
	id := uint16(1)
	negotiated := true
	dc, err := pc.CreateDataChannel("faces", &webrtc.DataChannelInit{
		ID:         &id,
		Negotiated: &negotiated,
	})
	if err != nil {
		return err
	}
	hub.add(dc)
	return nil
}

func newAPI() *webrtc.API {
	mediaEngine := &webrtc.MediaEngine{}
	videoRTCPFeedback := []webrtc.RTCPFeedback{
		{Type: "goog-remb", Parameter: ""},
		{Type: "ccm", Parameter: "fir"},
		{Type: "nack", Parameter: ""},
		{Type: "nack", Parameter: "pli"},
	}
	if err := mediaEngine.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: webrtc.RTPCodecCapability{
			MimeType:     webrtc.MimeTypeVP8,
			ClockRate:    90000,
			RTCPFeedback: videoRTCPFeedback,
		},
		PayloadType: 96,
	}, webrtc.RTPCodecTypeVideo); err != nil {
		panic(err)
	}
	if err := mediaEngine.RegisterCodec(webrtc.RTPCodecParameters{
		RTPCodecCapability: webrtc.RTPCodecCapability{
			MimeType:     webrtc.MimeTypeOpus,
			ClockRate:    48000,
			Channels:     2,
			SDPFmtpLine:  "minptime=10;useinbandfec=1",
		},
		PayloadType: 111,
	}, webrtc.RTPCodecTypeAudio); err != nil {
		panic(err)
	}

	interceptorRegistry := &interceptor.Registry{}
	if err := webrtc.RegisterDefaultInterceptors(mediaEngine, interceptorRegistry); err != nil {
		panic(err)
	}
	pli, err := intervalpli.NewReceiverInterceptor()
	if err != nil {
		panic(err)
	}
	interceptorRegistry.Add(pli)

	return webrtc.NewAPI(
		webrtc.WithMediaEngine(mediaEngine),
		webrtc.WithInterceptorRegistry(interceptorRegistry),
	)
}

// nolint:gocognit, cyclop
func main() {
	port := flag.Int("port", 8080, "http/https server port")
	// 127.0.0.1 = same PC only (no Windows Firewall popup).
	// 0.0.0.0 = LAN access (Firewall may ask once — prefer a built .exe, not go run).
	listenHost := flag.String("listen", "127.0.0.1", "listen address (127.0.0.1 or 0.0.0.0)")
	https := flag.Bool("https", false, "serve HTTPS (needed for camera on LAN IP)")
	flag.Parse()

	// Render / PaaS: public HTTPS is terminated at the proxy — app speaks plain HTTP on $PORT.
	cloud := os.Getenv("RENDER") == "true" || os.Getenv("LIVECAST_CLOUD") == "1"
	if p := strings.TrimSpace(os.Getenv("PORT")); p != "" {
		if n, err := strconv.Atoi(p); err == nil && n > 0 {
			*port = n
		}
	}
	if cloud {
		*listenHost = "0.0.0.0"
		*https = false
		fmt.Println("Cloud mode: HTTP on PORT (TLS at edge). Prefer TURN for WebRTC across NATs.")
	}

	// LAN listen almost always needs HTTPS for getUserMedia in Chrome/Edge.
	if !cloud && *listenHost == "0.0.0.0" && !*https {
		*https = true
		fmt.Println("LAN mode: enabling HTTPS so camera/mic work on http(s)://<PC-IP>")
	}

	events, err := newFaceEventLogger()
	if err != nil {
		panic(err)
	}
	defer events.close()

	if os.Getenv("ANSWER_AUTO") == "1" {
		if err := startLocalAnswerService(); err != nil {
			fmt.Printf("answer service: %v\n", err)
		}
	} else {
		fmt.Println("answer auto: off (set ANSWER_AUTO=1 to enable Q&A bot)")
	}
	_ = saveTranslateConfig(loadTranslateConfig())

	peerConnectionConfig := webrtc.Configuration{
		ICEServers: loadICEServers(),
	}
	api := newAPI()
	rooms := newRoomManager(api, peerConnectionConfig, events)

	addr := *listenHost + ":" + strconv.Itoa(*port)
	sdpChan := httpSDPServer(addr, rooms, *https)
	scheme := "http"
	if *https {
		scheme = "https"
	}
	fmt.Printf("Open %s://127.0.0.1:%d (listen %s)\n", scheme, *port, addr)
	fmt.Println("Multi-session: use ?room=<id> — max", maxRooms, "rooms")
	if *https {
		fmt.Println("HTTPS self-signed: browser will warn once — click Advanced → Proceed")
		fmt.Printf("LAN camera URL: https://<THIS-PC-IP>:%d\n", *port)
	}
	if !cloud && *listenHost == "127.0.0.1" {
		fmt.Println("Localhost only — Windows Firewall popup will not appear.")
		fmt.Println("For phone/LAN camera: broadcast.exe -port 8081 -listen 0.0.0.0")
	}

	for ex := range sdpChan {
		go rooms.handleSignaling(ex)
	}
}

func loadICEServers() []webrtc.ICEServer {
	servers := []webrtc.ICEServer{
		{URLs: []string{"stun:stun.l.google.com:19302"}},
		{URLs: []string{"stun:stun.cloudflare.com:3478"}},
	}
	turnURLs := strings.TrimSpace(os.Getenv("TURN_URLS"))
	turnUser := strings.TrimSpace(os.Getenv("TURN_USERNAME"))
	turnPass := strings.TrimSpace(os.Getenv("TURN_CREDENTIAL"))
	if turnURLs != "" && turnUser != "" && turnPass != "" {
		urls := []string{}
		for _, u := range strings.Split(turnURLs, ",") {
			u = strings.TrimSpace(u)
			if u != "" {
				urls = append(urls, u)
			}
		}
		if len(urls) > 0 {
			servers = append(servers, webrtc.ICEServer{
				URLs:       urls,
				Username:   turnUser,
				Credential: turnPass,
			})
			fmt.Printf("ICE: STUN + TURN (%d urls)\n", len(urls))
			return servers
		}
	}
	fmt.Println("ICE: STUN only — set TURN_URLS/TURN_USERNAME/TURN_CREDENTIAL for cross-network joins")
	return servers
}

func roomFromRequest(req *http.Request, rooms *RoomManager) *Room {
	id, err := normalizeRoomID(req.URL.Query().Get("room"))
	if err != nil {
		return nil
	}
	return rooms.get(id)
}

func handleViewer(
	api *webrtc.API,
	cfg webrtc.Configuration,
	hub *faceHub,
	transcripts *transcriptHub,
	answers *answerHub,
	translations *translateHub,
	localTracks []*webrtc.TrackLocalStaticRTP,
	viewer sdpExchange,
) error {
	recvOnlyOffer := webrtc.SessionDescription{}
	if err := decodeOffer(viewer.offer, &recvOnlyOffer); err != nil {
		return err
	}

	viewerPC, err := api.NewPeerConnection(cfg)
	if err != nil {
		return err
	}
	if err := createFacesChannel(viewerPC, hub); err != nil {
		_ = viewerPC.Close()
		return err
	}
	if err := createTranscriptChannel(viewerPC, transcripts); err != nil {
		_ = viewerPC.Close()
		return err
	}
	if err := createAnswerChannel(viewerPC, answers); err != nil {
		_ = viewerPC.Close()
		return err
	}
	if err := createTranslateChannel(viewerPC, translations); err != nil {
		_ = viewerPC.Close()
		return err
	}

	for _, localTrack := range localTracks {
		rtpSender, err := viewerPC.AddTrack(localTrack)
		if err != nil {
			_ = viewerPC.Close()
			return err
		}
		go func(sender *webrtc.RTPSender) {
			rtcpBuf := make([]byte, 1500)
			for {
				if _, _, rtcpErr := sender.Read(rtcpBuf); rtcpErr != nil {
					return
				}
			}
		}(rtpSender)
	}

	if err := viewerPC.SetRemoteDescription(recvOnlyOffer); err != nil {
		_ = viewerPC.Close()
		return err
	}
	answer, err := viewerPC.CreateAnswer(nil)
	if err != nil {
		_ = viewerPC.Close()
		return err
	}
	gatherComplete := webrtc.GatheringCompletePromise(viewerPC)
	if err := viewerPC.SetLocalDescription(answer); err != nil {
		_ = viewerPC.Close()
		return err
	}
	select {
	case <-gatherComplete:
	case <-time.After(10 * time.Second):
	}

	viewer.answer <- encode(viewerPC.LocalDescription())
	fmt.Println("Viewer answer ready")
	return nil
}

func forwardAudio(remoteTrack *webrtc.TrackRemote, localTrack *webrtc.TrackLocalStaticRTP, asr *whisperWorker) {
	n := 0
	fmt.Printf("forwardAudio: start mime=%s asr=%v\n", remoteTrack.Codec().MimeType, asr != nil)
	for {
		pkt, _, readErr := remoteTrack.ReadRTP()
		if readErr != nil {
			fmt.Printf("forwardAudio: read end after %d pkts: %v\n", n, readErr)
			return
		}
		n++
		payload := pkt.Payload
		// Strip one-byte RTP padding if present is already handled by pion; still guard.
		if asr != nil && len(payload) > 0 {
			asr.submit(append([]byte(nil), payload...))
		}
		if n == 1 || n%100 == 0 {
			fmt.Printf("forwardAudio: pkt #%d payload=%d\n", n, len(payload))
		}
		// Fan-out to viewers; do not stop ASR if a viewer pipe fails.
		if writeErr := localTrack.WriteRTP(pkt); writeErr != nil && !errors.Is(writeErr, io.ErrClosedPipe) {
			fmt.Printf("forwardAudio: write warning: %v (continuing ASR)\n", writeErr)
		}
	}
}

func forwardVideoWithFaceDetect(
	remoteTrack *webrtc.TrackRemote,
	localTrack *webrtc.TrackLocalStaticRTP,
	worker *faceWorker,
) {
	builder := samplebuilder.New(20, &codecs.VP8Packet{}, 90000)
	for {
		pkt, _, readErr := remoteTrack.ReadRTP()
		if readErr != nil {
			return
		}
		if writeErr := localTrack.WriteRTP(pkt); writeErr != nil && !errors.Is(writeErr, io.ErrClosedPipe) {
			return
		}
		builder.Push(pkt)
		for {
			sample := builder.Pop()
			if sample == nil {
				break
			}
			frame := append([]byte(nil), sample.Data...)
			worker.submit(frame)
		}
	}
}

func encode(obj *webrtc.SessionDescription) string {
	b, err := json.Marshal(obj)
	if err != nil {
		panic(err)
	}
	return base64.StdEncoding.EncodeToString(b)
}

func decodeOffer(in string, obj *webrtc.SessionDescription) error {
	b, err := base64.StdEncoding.DecodeString(in)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, obj)
}

func httpSDPServer(addr string, rooms *RoomManager, useHTTPS bool) chan sdpExchange {
	sdpChan := make(chan sdpExchange)

	submitOffer := func(offer, role, room string) (string, string) {
		exchange := sdpExchange{
			offer:  offer,
			role:   role,
			room:   room,
			answer: make(chan string, 1),
			err:    make(chan string, 1),
		}
		select {
		case sdpChan <- exchange:
		case <-time.After(2 * time.Second):
			return "", "server busy — wait and retry"
		}
		select {
		case answer := <-exchange.answer:
			return answer, ""
		case msg := <-exchange.err:
			return "", msg
		case <-time.After(60 * time.Second):
			return "", "signaling timeout — refresh and try once"
		}
	}

	writeJSON := func(res http.ResponseWriter, b []byte) {
		res.Header().Set("Access-Control-Allow-Origin", "*")
		res.Header().Set("Content-Type", "application/json")
		res.Header().Set("Cache-Control", "no-store")
		_, _ = res.Write(b)
	}

	mux := http.NewServeMux()
	mux.Handle("/", http.FileServer(http.Dir("web")))
	mux.HandleFunc("/rooms", func(res http.ResponseWriter, req *http.Request) {
		b, _ := json.Marshal(map[string]any{"rooms": rooms.list(), "max": maxRooms})
		writeJSON(res, b)
	})
	mux.HandleFunc("/room/new", func(res http.ResponseWriter, req *http.Request) {
		res.Header().Set("Access-Control-Allow-Origin", "*")
		if req.Method != http.MethodPost && req.Method != http.MethodGet {
			http.Error(res, "GET or POST", http.StatusMethodNotAllowed)
			return
		}
		id := newRoomID()
		b, _ := json.Marshal(map[string]string{"room": id})
		writeJSON(res, b)
	})
	mux.HandleFunc("/face", func(res http.ResponseWriter, req *http.Request) {
		if r := roomFromRequest(req, rooms); r != nil {
			writeJSON(res, r.snapshotFace())
			return
		}
		writeJSON(res, []byte(`{"present":false,"count":0,"status":"pick a room","confidence":0}`))
	})
	mux.HandleFunc("/transcript", func(res http.ResponseWriter, req *http.Request) {
		if r := roomFromRequest(req, rooms); r != nil {
			writeJSON(res, r.snapshotTranscript())
			return
		}
		writeJSON(res, []byte(`{"text":"","status":"pick a room","partial":false}`))
	})
	mux.HandleFunc("/answer", func(res http.ResponseWriter, req *http.Request) {
		if r := roomFromRequest(req, rooms); r != nil {
			writeJSON(res, r.snapshotAnswer())
			return
		}
		writeJSON(res, []byte(`{"question":"","answer":"","status":"pick a room"}`))
	})
	mux.HandleFunc("/translate", func(res http.ResponseWriter, req *http.Request) {
		if r := roomFromRequest(req, rooms); r != nil {
			writeJSON(res, r.snapshotTranslate())
			return
		}
		writeJSON(res, []byte(`{"source":"","translated":"","status":"pick a room"}`))
	})
	mux.HandleFunc("/translate-config", func(res http.ResponseWriter, req *http.Request) {
		res.Header().Set("Access-Control-Allow-Origin", "*")
		res.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		res.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if req.Method == http.MethodOptions {
			res.WriteHeader(http.StatusOK)
			return
		}
		if req.Method == http.MethodGet {
			cfg := loadTranslateConfig()
			b, _ := json.Marshal(cfg)
			writeJSON(res, b)
			return
		}
		if req.Method != http.MethodPost {
			http.Error(res, "GET or POST only", http.StatusMethodNotAllowed)
			return
		}
		var cfg translateConfig
		if err := json.NewDecoder(req.Body).Decode(&cfg); err != nil {
			http.Error(res, "invalid json", http.StatusBadRequest)
			return
		}
		if err := saveTranslateConfig(cfg); err != nil {
			http.Error(res, err.Error(), http.StatusInternalServerError)
			return
		}
		b, _ := json.Marshal(cfg)
		writeJSON(res, b)
		fmt.Printf("translate-config: target=%s enabled=%v\n", cfg.TargetLang, cfg.Enabled)
	})
	mux.HandleFunc("/config", func(res http.ResponseWriter, req *http.Request) {
		res.Header().Set("Access-Control-Allow-Origin", "*")
		res.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		res.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if req.Method == http.MethodOptions {
			res.WriteHeader(http.StatusOK)
			return
		}
		cfgPath := filepath.Join("facedetect", "config.json")
		if req.Method == http.MethodGet {
			b, err := os.ReadFile(cfgPath)
			if err != nil {
				b = []byte(`{"window_sec":300}`)
			}
			writeJSON(res, b)
			return
		}
		if req.Method != http.MethodPost {
			http.Error(res, "GET or POST only", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			WindowSec int `json:"window_sec"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(res, "invalid json", http.StatusBadRequest)
			return
		}
		if body.WindowSec < 30 {
			body.WindowSec = 30
		}
		if body.WindowSec > 86400 {
			body.WindowSec = 86400
		}
		out, _ := json.Marshal(map[string]int{"window_sec": body.WindowSec})
		if err := os.WriteFile(cfgPath, out, 0o644); err != nil {
			http.Error(res, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(res, out)
		fmt.Printf("config: window_sec=%d\n", body.WindowSec)
	})
	mux.HandleFunc("/sdp", func(res http.ResponseWriter, req *http.Request) {
		res.Header().Set("Access-Control-Allow-Origin", "*")
		if req.Method == http.MethodOptions {
			res.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
			res.WriteHeader(http.StatusOK)
			return
		}
		if req.Method != http.MethodPost {
			http.Error(res, "POST only", http.StatusMethodNotAllowed)
			return
		}
		var body struct {
			SDP  string `json:"sdp"`
			Role string `json:"role"`
			Room string `json:"room"`
		}
		raw, err := io.ReadAll(req.Body)
		if err != nil {
			http.Error(res, err.Error(), http.StatusBadRequest)
			return
		}
		if err := json.Unmarshal(raw, &body); err != nil || body.SDP == "" {
			body.SDP = string(raw)
			body.Role = "publisher"
			body.Room = "default"
		}
		answer, errMsg := submitOffer(body.SDP, body.Role, body.Room)
		if errMsg != "" {
			http.Error(res, errMsg, http.StatusConflict)
			return
		}
		_, _ = fmt.Fprint(res, answer)
	})

	mux.Handle("/ws", websocket.Handler(func(ws *websocket.Conn) {
		defer func() { _ = ws.Close() }()
		for {
			var msg struct {
				Type string `json:"type"`
				Role string `json:"role"`
				Room string `json:"room"`
				SDP  string `json:"sdp"`
			}
			if err := websocket.JSON.Receive(ws, &msg); err != nil {
				return
			}
			if msg.Type != "offer" || msg.SDP == "" {
				_ = websocket.JSON.Send(ws, map[string]string{
					"type":    "error",
					"message": "expected {type:offer, room, role, sdp}",
				})
				continue
			}
			answer, errMsg := submitOffer(msg.SDP, msg.Role, msg.Room)
			if errMsg != "" {
				_ = websocket.JSON.Send(ws, map[string]string{"type": "error", "message": errMsg})
				continue
			}
			_ = websocket.JSON.Send(ws, map[string]string{
				"type": "answer",
				"role": msg.Role,
				"room": msg.Room,
				"sdp":  answer,
			})
		}
	}))

	go func() {
		if useHTTPS {
			certFile, keyFile, err := ensureDevTLS(filepath.Join("certs"))
			if err != nil {
				panic(err)
			}
			fmt.Println("Serving HTTPS with", certFile)
			// nolint: gosec
			panic(http.ListenAndServeTLS(addr, certFile, keyFile, mux))
		}
		// nolint: gosec
		panic(http.ListenAndServe(addr, mux))
	}()

	return sdpChan
}
