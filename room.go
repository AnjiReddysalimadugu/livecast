// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

//go:build !js

package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/pion/webrtc/v4"
)

const maxRooms = 5

var roomIDPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{3,24}$`)

type sdpExchange struct {
	offer  string
	role   string // publisher | viewer
	room   string
	answer chan string
	err    chan string
}

type Room struct {
	ID string

	mu          sync.Mutex
	publisherPC *webrtc.PeerConnection
	tracks      []*webrtc.TrackLocalStaticRTP
	ready       chan struct{} // closed when tracks ready
	closed      bool

	faces        *faceHub
	transcripts  *transcriptHub
	answers      *answerHub
	translations *translateHub

	faceWorker *faceWorker
	asr        *whisperWorker
	translator *liveTranslator
	answerBot  *answerClient
}

type RoomManager struct {
	mu     sync.Mutex
	rooms  map[string]*Room
	api    *webrtc.API
	cfg    webrtc.Configuration
	events *faceEventLogger
}

func newRoomManager(api *webrtc.API, cfg webrtc.Configuration, events *faceEventLogger) *RoomManager {
	return &RoomManager{
		rooms:  make(map[string]*Room),
		api:    api,
		cfg:    cfg,
		events: events,
	}
}

func normalizeRoomID(id string) (string, error) {
	id = strings.ToLower(strings.TrimSpace(id))
	if id == "" {
		return "", fmt.Errorf("room id required")
	}
	if !roomIDPattern.MatchString(id) {
		return "", fmt.Errorf("room id must be 3–24 chars: letters, numbers, _-")
	}
	return id, nil
}

func newRoomID() string {
	var b [4]byte
	_, _ = rand.Read(b[:])
	return hex.EncodeToString(b[:])
}

func (rm *RoomManager) list() []map[string]any {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	out := make([]map[string]any, 0, len(rm.rooms))
	for id, r := range rm.rooms {
		r.mu.Lock()
		hasPub := r.publisherPC != nil && !r.closed
		nTracks := len(r.tracks)
		r.mu.Unlock()
		out = append(out, map[string]any{
			"id":         id,
			"publisher":  hasPub,
			"track_count": nTracks,
		})
	}
	return out
}

func (rm *RoomManager) get(id string) *Room {
	rm.mu.Lock()
	defer rm.mu.Unlock()
	return rm.rooms[id]
}

func (rm *RoomManager) getOrCreate(id string) (*Room, error) {
	id, err := normalizeRoomID(id)
	if err != nil {
		return nil, err
	}
	rm.mu.Lock()
	defer rm.mu.Unlock()
	if r := rm.rooms[id]; r != nil {
		return r, nil
	}
	if len(rm.rooms) >= maxRooms {
		return nil, fmt.Errorf("too many rooms (max %d) — close an idle room", maxRooms)
	}
	r := &Room{
		ID:           id,
		faces:        &faceHub{events: rm.events},
		transcripts:  &transcriptHub{},
		answers:      &answerHub{},
		translations: &translateHub{},
		ready:        make(chan struct{}),
	}
	r.translator = newLiveTranslator(r.translations)
	r.answerBot = newAnswerClient(r.answers)

	fw, err := startFaceWorker(r.faces)
	if err != nil {
		return nil, fmt.Errorf("face worker: %w", err)
	}
	r.faceWorker = fw

	asr, err := startWhisperWorker(r.transcripts, func(line []byte) {
		r.translator.onTranscriptJSON(line)
		if os.Getenv("ANSWER_AUTO") == "1" {
			r.answerBot.onTranscriptJSON(line)
		}
	})
	if err != nil {
		fmt.Printf("room %s: whisper disabled: %v\n", id, err)
	} else {
		r.asr = asr
	}

	rm.rooms[id] = r
	fmt.Printf("room created: %s (%d/%d)\n", id, len(rm.rooms), maxRooms)
	return r, nil
}

func (rm *RoomManager) remove(id string) {
	rm.mu.Lock()
	r := rm.rooms[id]
	delete(rm.rooms, id)
	rm.mu.Unlock()
	if r != nil {
		r.shutdown()
		fmt.Printf("room removed: %s\n", id)
	}
}

func (r *Room) shutdown() {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.closed = true
	pc := r.publisherPC
	r.publisherPC = nil
	r.tracks = nil
	fw := r.faceWorker
	asr := r.asr
	r.faceWorker = nil
	r.asr = nil
	r.mu.Unlock()
	if pc != nil {
		_ = pc.Close()
	}
	fw.Close()
	asr.Close()
}

func (r *Room) snapshotFace() []byte       { return r.faces.snapshot() }
func (r *Room) snapshotTranscript() []byte { return r.transcripts.snapshot() }
func (r *Room) snapshotAnswer() []byte     { return r.answers.snapshot() }
func (r *Room) snapshotTranslate() []byte  { return r.translations.snapshot() }

func (rm *RoomManager) handleSignaling(ex sdpExchange) {
	role := strings.ToLower(strings.TrimSpace(ex.role))
	if role == "" {
		role = "publisher"
	}
	roomID := strings.TrimSpace(ex.room)
	if roomID == "" {
		ex.err <- "room id required"
		return
	}

	switch role {
	case "publisher":
		rm.publish(ex)
	case "viewer":
		rm.view(ex)
	default:
		ex.err <- "role must be publisher or viewer"
	}
}

func (rm *RoomManager) publish(ex sdpExchange) {
	r, err := rm.getOrCreate(ex.room)
	if err != nil {
		ex.err <- err.Error()
		return
	}

	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		ex.err <- "room closed — create a new room"
		return
	}
	// Replace previous publisher in this room.
	if r.publisherPC != nil {
		_ = r.publisherPC.Close()
		r.publisherPC = nil
		r.tracks = nil
		select {
		case <-r.ready:
		default:
		}
		r.ready = make(chan struct{})
		r.faces.reset()
		r.transcripts.reset()
		r.answers.reset()
		r.translations.reset()
	}
	r.mu.Unlock()

	offer := webrtc.SessionDescription{}
	if err := decodeOffer(ex.offer, &offer); err != nil {
		ex.err <- "invalid offer SDP"
		return
	}

	pc, err := rm.api.NewPeerConnection(rm.cfg)
	if err != nil {
		ex.err <- err.Error()
		return
	}

	if err := createFacesChannel(pc, r.faces); err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}
	if err := createTranscriptChannel(pc, r.transcripts); err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}
	if err := createAnswerChannel(pc, r.answers); err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}
	if err := createTranslateChannel(pc, r.translations); err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}
	// Match typical browser offer order (audio then video). Wrong order can
	// negotiate video but leave audio with zero RTP — Whisper then stays at pkt:0.
	recv := webrtc.RTPTransceiverInit{Direction: webrtc.RTPTransceiverDirectionRecvonly}
	if _, err = pc.AddTransceiverFromKind(webrtc.RTPCodecTypeAudio, recv); err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}
	if _, err = pc.AddTransceiverFromKind(webrtc.RTPCodecTypeVideo, recv); err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}

	localTrackChan := make(chan *webrtc.TrackLocalStaticRTP, 2)
	pc.OnTrack(func(remoteTrack *webrtc.TrackRemote, receiver *webrtc.RTPReceiver) { //nolint: revive
		trackID := "video"
		if remoteTrack.Kind() == webrtc.RTPCodecTypeAudio {
			trackID = "audio"
		}
		localTrack, newTrackErr := webrtc.NewTrackLocalStaticRTP(
			remoteTrack.Codec().RTPCodecCapability, trackID, "pion",
		)
		if newTrackErr != nil {
			fmt.Printf("room %s track create error: %v\n", r.ID, newTrackErr)
			return
		}
		select {
		case localTrackChan <- localTrack:
		case <-time.After(2 * time.Second):
			fmt.Printf("room %s: dropped late %s track\n", r.ID, trackID)
		}
		fmt.Printf("room %s: publisher %s (%s)\n", r.ID, trackID, remoteTrack.Codec().MimeType)

		// Keep RTCP flowing (helps some browsers keep sending media).
		if receiver != nil {
			go func(rcvr *webrtc.RTPReceiver) {
				buf := make([]byte, 1500)
				for {
					if _, _, err := rcvr.Read(buf); err != nil {
						return
					}
				}
			}(receiver)
		}

		if remoteTrack.Kind() == webrtc.RTPCodecTypeVideo {
			go forwardVideoWithFaceDetect(remoteTrack, localTrack, r.faceWorker)
			return
		}
		asr := r.asr
		if asr == nil {
			fmt.Printf("room %s: WARNING audio track but whisper worker is nil\n", r.ID)
		} else {
			fmt.Printf("room %s: starting whisper audio tap\n", r.ID)
		}
		go forwardAudio(remoteTrack, localTrack, asr)
	})

	if err := pc.SetRemoteDescription(offer); err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}
	answer, err := pc.CreateAnswer(nil)
	if err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}
	gatherComplete := webrtc.GatheringCompletePromise(pc)
	if err := pc.SetLocalDescription(answer); err != nil {
		_ = pc.Close()
		ex.err <- err.Error()
		return
	}
	select {
	case <-gatherComplete:
	case <-time.After(10 * time.Second):
	}

	r.mu.Lock()
	r.publisherPC = pc
	r.mu.Unlock()

	ex.answer <- encode(pc.LocalDescription())
	fmt.Printf("room %s: publisher answer — waiting for tracks…\n", r.ID)

	tracks, ok := waitTracksOnly(localTrackChan, 45*time.Second)
	if !ok {
		fmt.Printf("room %s: publisher media timeout\n", r.ID)
		_ = pc.Close()
		rm.remove(r.ID)
		return
	}

	r.mu.Lock()
	r.tracks = tracks
	ready := r.ready
	r.mu.Unlock()
	select {
	case <-ready:
	default:
		close(ready)
	}
	fmt.Printf("room %s: media ready (%d tracks)\n", r.ID, len(tracks))

	pc.OnICEConnectionStateChange(func(s webrtc.ICEConnectionState) {
		fmt.Printf("room %s publisher ICE: %s\n", r.ID, s.String())
		// "disconnected" is often a brief blip (esp. on cloud). Only tear down on failed/closed.
		if s != webrtc.ICEConnectionStateFailed && s != webrtc.ICEConnectionStateClosed {
			return
		}
		go func() {
			time.Sleep(8 * time.Second)
			r.mu.Lock()
			alive := r.publisherPC == pc
			r.mu.Unlock()
			if !alive {
				return
			}
			st := pc.ICEConnectionState()
			if st == webrtc.ICEConnectionStateFailed || st == webrtc.ICEConnectionStateClosed {
				fmt.Printf("room %s: publisher gone — closing room\n", r.ID)
				rm.remove(r.ID)
			}
		}()
	})
}

func (rm *RoomManager) view(ex sdpExchange) {
	id, err := normalizeRoomID(ex.room)
	if err != nil {
		ex.err <- err.Error()
		return
	}
	r := rm.get(id)
	if r == nil {
		ex.err <- "room not found — create/publish first"
		return
	}

	// Wait briefly for publisher tracks.
	deadline := time.After(15 * time.Second)
	for {
		r.mu.Lock()
		tracks := append([]*webrtc.TrackLocalStaticRTP(nil), r.tracks...)
		ready := r.ready
		closed := r.closed
		r.mu.Unlock()
		if closed {
			ex.err <- "room closed"
			return
		}
		if len(tracks) > 0 {
			if err := handleViewer(rm.api, rm.cfg, r.faces, r.transcripts, r.answers, r.translations, tracks, ex); err != nil {
				fmt.Printf("room %s viewer error: %v\n", r.ID, err)
				ex.err <- err.Error()
			}
			return
		}
		select {
		case <-ready:
			continue
		case <-deadline:
			ex.err <- "no publisher media in this room yet — wait and Join again"
			return
		case <-time.After(200 * time.Millisecond):
		}
	}
}

// waitTracksOnly waits for publisher tracks without a shared sdpChan.
func waitTracksOnly(localTrackChan <-chan *webrtc.TrackLocalStaticRTP, timeout time.Duration) ([]*webrtc.TrackLocalStaticRTP, bool) {
	tracks := make([]*webrtc.TrackLocalStaticRTP, 0, 2)
	deadline := time.After(timeout)
	for len(tracks) < 2 {
		select {
		case t := <-localTrackChan:
			tracks = append(tracks, t)
			if len(tracks) >= 1 {
				select {
				case t2 := <-localTrackChan:
					tracks = append(tracks, t2)
				case <-time.After(3 * time.Second):
				}
				return tracks, true
			}
		case <-deadline:
			if len(tracks) > 0 {
				return tracks, true
			}
			return nil, false
		}
	}
	return tracks, true
}
