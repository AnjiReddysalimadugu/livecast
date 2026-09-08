// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

//go:build !js

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/pion/webrtc/v4"
)

type answerHub struct {
	mu    sync.Mutex
	last  []byte
	chans []*webrtc.DataChannel
}

func (h *answerHub) snapshot() []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.last) == 0 {
		return []byte(`{"question":"","answer":"","status":"waiting"}`)
	}
	return append([]byte(nil), h.last...)
}

func (h *answerHub) reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.chans = nil
}

func (h *answerHub) add(dc *webrtc.DataChannel) {
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

func (h *answerHub) broadcast(msg []byte) {
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

type answerClient struct {
	url    string
	hub    *answerHub
	client *http.Client

	mu       sync.Mutex
	lastQ    string
	inflight bool
}

func answerURL() string {
	if u := strings.TrimSpace(os.Getenv("ANSWER_URL")); u != "" {
		return u
	}
	return "http://127.0.0.1:8091/answer"
}

func startLocalAnswerService() error {
	// If ANSWER_URL points elsewhere, do not spawn local process.
	if u := strings.TrimSpace(os.Getenv("ANSWER_URL")); u != "" &&
		!strings.Contains(u, "127.0.0.1") && !strings.Contains(u, "localhost") {
		fmt.Printf("answer: using remote service %s\n", u)
		return nil
	}
	script, err := filepath.Abs(filepath.Join("answer", "server.py"))
	if err != nil {
		return err
	}
	if _, err := os.Stat(script); err != nil {
		return err
	}
	cmd := exec.Command("python", "-u", script, "--host", "127.0.0.1", "--port", "8091")
	cmd.Stderr = os.Stderr
	cmd.Stdout = os.Stderr
	if err := cmd.Start(); err != nil {
		return err
	}
	go func() { _ = cmd.Wait() }()

	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		res, err := http.Get("http://127.0.0.1:8091/health")
		if err == nil {
			_ = res.Body.Close()
			if res.StatusCode == 200 {
				fmt.Println("answer-service: local Ollama Q&A ready on :8091")
				return nil
			}
		}
		time.Sleep(400 * time.Millisecond)
	}
	return fmt.Errorf("answer service health timeout")
}

func newAnswerClient(hub *answerHub) *answerClient {
	return &answerClient{
		url: answerURL(),
		hub: hub,
		client: &http.Client{
			Timeout: 90 * time.Second,
		},
	}
}

func looksLikeQuestion(text string) bool {
	t := strings.TrimSpace(text)
	if len(t) < 8 {
		return false
	}
	letters := 0
	for _, r := range t {
		if unicode.IsLetter(r) {
			letters++
		}
	}
	return letters >= 4
}

func (c *answerClient) onTranscriptJSON(raw []byte) {
	if c == nil || c.hub == nil {
		return
	}
	var payload struct {
		Text   string `json:"text"`
		Status string `json:"status"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return
	}
	q := strings.TrimSpace(payload.Text)
	if payload.Status != "ok" || !looksLikeQuestion(q) {
		return
	}

	c.mu.Lock()
	if q == c.lastQ || c.inflight {
		c.mu.Unlock()
		return
	}
	c.lastQ = q
	c.inflight = true
	c.mu.Unlock()

	go func(question string) {
		defer func() {
			c.mu.Lock()
			c.inflight = false
			c.mu.Unlock()
		}()

		pending, _ := json.Marshal(map[string]any{
			"question": question,
			"answer":   "",
			"status":   "thinking",
		})
		c.hub.broadcast(pending)
		fmt.Printf("answer: asking LLM for %q\n", question)

		body, _ := json.Marshal(map[string]string{"question": question})
		req, err := http.NewRequest(http.MethodPost, c.url, bytes.NewReader(body))
		if err != nil {
			c.emitError(question, err.Error())
			return
		}
		req.Header.Set("Content-Type", "application/json")
		res, err := c.client.Do(req)
		if err != nil {
			c.emitError(question, "answer service unreachable: "+err.Error())
			return
		}
		defer res.Body.Close()
		raw, _ := io.ReadAll(res.Body)
		if res.StatusCode >= 300 {
			c.emitError(question, fmt.Sprintf("HTTP %d: %s", res.StatusCode, string(raw)))
			return
		}
		c.hub.broadcast(raw)
		fmt.Printf("answer: %s\n", raw)
	}(q)
}

func (c *answerClient) emitError(question, msg string) {
	out, _ := json.Marshal(map[string]any{
		"question": question,
		"answer":   "",
		"status":   "error",
		"error":    msg,
	})
	c.hub.broadcast(out)
	fmt.Printf("answer error: %s\n", msg)
}

func createAnswerChannel(pc *webrtc.PeerConnection, hub *answerHub) error {
	if hub == nil {
		return nil
	}
	id := uint16(3)
	negotiated := true
	dc, err := pc.CreateDataChannel("answers", &webrtc.DataChannelInit{
		ID:         &id,
		Negotiated: &negotiated,
	})
	if err != nil {
		return err
	}
	hub.add(dc)
	return nil
}
