// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

//go:build !js

package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/pion/webrtc/v4"
)

func getenvDefault(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

type transcriptHub struct {
	mu    sync.Mutex
	last  []byte
	chans []*webrtc.DataChannel
}

func (h *transcriptHub) snapshot() []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.last) == 0 {
		return []byte(`{"text":"","status":"waiting for audio","partial":false}`)
	}
	return append([]byte(nil), h.last...)
}

func (h *transcriptHub) reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.chans = nil
}

func (h *transcriptHub) add(dc *webrtc.DataChannel) {
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

func (h *transcriptHub) broadcast(msg []byte) {
	h.mu.Lock()
	h.last = append([]byte(nil), msg...)
	chans := append([]*webrtc.DataChannel(nil), h.chans...)
	h.mu.Unlock()

	for _, dc := range chans {
		if dc == nil || dc.ReadyState() != webrtc.DataChannelStateOpen {
			continue
		}
		_ = dc.SendText(string(msg))
	}
}

type whisperWorker struct {
	stdin  io.WriteCloser
	cmd    *exec.Cmd
	mu     sync.Mutex
	frames chan []byte
	once   sync.Once
}

func startWhisperWorker(hub *transcriptHub, onLine func([]byte)) (*whisperWorker, error) {
	script, err := filepath.Abs(filepath.Join("whisper", "worker.py"))
	if err != nil {
		return nil, err
	}
	if _, err := os.Stat(script); err != nil {
		return nil, err
	}
	cmd := exec.Command("python", "-u", script)
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"WHISPER_MODEL="+getenvDefault("WHISPER_MODEL", "tiny"),
		"WHISPER_LANGUAGE="+getenvDefault("WHISPER_LANGUAGE", "en"),
		"WHISPER_CHUNK_SEC="+getenvDefault("WHISPER_CHUNK_SEC", "1.5"),
		"WHISPER_STEP_SEC="+getenvDefault("WHISPER_STEP_SEC", "0.75"),
		"WHISPER_MIN_INTERVAL="+getenvDefault("WHISPER_MIN_INTERVAL", "0.12"),
		"WHISPER_MAX_BUF_SEC="+getenvDefault("WHISPER_MAX_BUF_SEC", "4.5"),
		"WHISPER_MAX_LAG_SEC="+getenvDefault("WHISPER_MAX_LAG_SEC", "2.2"),
	)
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

	w := &whisperWorker{stdin: stdin, cmd: cmd, frames: make(chan []byte, 256)}

	go func() {
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for sc.Scan() {
			line := append([]byte(nil), sc.Bytes()...)
			hub.broadcast(line)
			fmt.Printf("whisper: %s\n", line)
			if onLine != nil {
				onLine(line)
			}
		}
		if err := sc.Err(); err != nil {
			fmt.Printf("whisper stdout err: %v\n", err)
		}
	}()

	go func() {
		written := 0
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
				fmt.Printf("whisper-worker write error after %d frames: %v\n", written, err)
				return
			}
			written++
			if written == 1 || written%100 == 0 {
				fmt.Printf("whisper-worker: wrote %d opus frames to python\n", written)
			}
		}
	}()

	fmt.Println("Server-side Whisper STT worker started (audio stream → text)")
	return w, nil
}

func (w *whisperWorker) submit(frame []byte) {
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
			fmt.Println("whisper-worker: frame queue full — dropping Opus packet")
		}
	}
}

func (w *whisperWorker) Close() {
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

func createTranscriptChannel(pc *webrtc.PeerConnection, hub *transcriptHub) error {
	if hub == nil {
		return nil
	}
	id := uint16(2)
	negotiated := true
	dc, err := pc.CreateDataChannel("transcripts", &webrtc.DataChannelInit{
		ID:         &id,
		Negotiated: &negotiated,
	})
	if err != nil {
		return err
	}
	hub.add(dc)
	return nil
}
