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
	h.last = nil
}

func (h *answerHub) add(dc *webrtc.DataChannel) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.chans = pruneOpenDCs(h.chans)
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
	h.chans = pruneOpenDCs(h.chans)
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
	lastAt   time.Time
	inflight bool
}

func voiceAssistEnabled() bool {
	v := strings.TrimSpace(os.Getenv("VOICE_ASSIST"))
	if v == "0" || strings.EqualFold(v, "false") || strings.EqualFold(v, "off") {
		return false
	}
	// Default ON — ChatGPT-style voice reply loop (Whisper → LLM → browser TTS).
	if v == "" {
		return true
	}
	return v == "1" || strings.EqualFold(v, "true") || strings.EqualFold(v, "on")
}

func answerURL() string {
	if u := strings.TrimSpace(os.Getenv("ANSWER_URL")); u != "" {
		return u
	}
	return "http://127.0.0.1:8091/answer"
}

func startLocalAnswerService() error {
	if u := strings.TrimSpace(os.Getenv("ANSWER_URL")); u != "" &&
		!strings.Contains(u, "127.0.0.1") && !strings.Contains(u, "localhost") {
		fmt.Printf("answer: using remote service %s\n", u)
		return nil
	}
	// OpenAI path is in-process — no local Python needed.
	if strings.TrimSpace(os.Getenv("OPENAI_API_KEY")) != "" {
		fmt.Println("answer: OpenAI API key set — in-process voice replies")
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
				fmt.Println("answer-service: local Q&A ready on :8091")
				return nil
			}
		}
		time.Sleep(400 * time.Millisecond)
	}
	return fmt.Errorf("answer service health timeout (optional — OpenAI fallback/echo still works)")
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

func looksLikeUtterance(text string) bool {
	t := strings.TrimSpace(text)
	if len(t) < 6 {
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
	if c == nil || c.hub == nil || !voiceAssistEnabled() {
		return
	}
	var payload struct {
		Text    string `json:"text"`
		Status  string `json:"status"`
		Partial bool   `json:"partial"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return
	}
	if payload.Partial {
		return
	}
	q := strings.TrimSpace(payload.Text)
	if payload.Status != "ok" || !looksLikeUtterance(q) {
		return
	}

	c.mu.Lock()
	if q == c.lastQ || c.inflight || time.Since(c.lastAt) < 4*time.Second {
		c.mu.Unlock()
		return
	}
	c.lastQ = q
	c.lastAt = time.Now()
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
			"speak":    true,
		})
		c.hub.broadcast(pending)
		fmt.Printf("voice-assist: thinking for %q\n", question)

		answer, model, err := c.generateAnswer(question)
		if err != nil {
			c.emitError(question, err.Error())
			return
		}
		out, _ := json.Marshal(map[string]any{
			"question": question,
			"answer":   answer,
			"model":    model,
			"status":   "ok",
			"speak":    true,
		})
		c.hub.broadcast(out)
		fmt.Printf("voice-assist: %s\n", answer)
	}(q)
}

func (c *answerClient) generateAnswer(question string) (answer, model string, err error) {
	if key := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); key != "" {
		return askOpenAI(c.client, key, question)
	}
	// Prefer local/remote answer HTTP service (Ollama wrapper).
	body, _ := json.Marshal(map[string]string{"question": question})
	req, err := http.NewRequest(http.MethodPost, c.url, bytes.NewReader(body))
	if err != nil {
		return fallbackAnswer(question), "echo", nil
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := c.client.Do(req)
	if err != nil {
		return fallbackAnswer(question), "echo", nil
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)
	if res.StatusCode >= 300 {
		return fallbackAnswer(question), "echo", nil
	}
	var parsed struct {
		Answer string `json:"answer"`
		Model  string `json:"model"`
		Status string `json:"status"`
		Error  string `json:"error"`
	}
	if json.Unmarshal(raw, &parsed) != nil || strings.TrimSpace(parsed.Answer) == "" {
		return fallbackAnswer(question), "echo", nil
	}
	model = parsed.Model
	if model == "" {
		model = "answer-service"
	}
	return strings.TrimSpace(parsed.Answer), model, nil
}

func fallbackAnswer(question string) string {
	return "I heard: " + question + ". Set OPENAI_API_KEY for smarter voice replies."
}

func askOpenAI(client *http.Client, apiKey, question string) (string, string, error) {
	base := strings.TrimRight(strings.TrimSpace(os.Getenv("OPENAI_BASE_URL")), "/")
	if base == "" {
		base = "https://api.openai.com/v1"
	}
	model := strings.TrimSpace(os.Getenv("OPENAI_MODEL"))
	if model == "" {
		model = "gpt-4o-mini"
	}
	system := strings.TrimSpace(os.Getenv("ANSWER_SYSTEM"))
	if system == "" {
		system = "You are a helpful live-call voice assistant. Reply in 1-3 short sentences, same language as the user. No markdown."
	}
	payload := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "system", "content": system},
			{"role": "user", "content": question},
		},
		"temperature": 0.5,
		"max_tokens":  180,
	}
	body, _ := json.Marshal(payload)
	req, err := http.NewRequest(http.MethodPost, base+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", model, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	res, err := client.Do(req)
	if err != nil {
		return "", model, err
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)
	if res.StatusCode >= 300 {
		return "", model, fmt.Errorf("openai HTTP %d: %s", res.StatusCode, string(raw))
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", model, err
	}
	if len(parsed.Choices) == 0 {
		return "", model, fmt.Errorf("openai empty choices")
	}
	return strings.TrimSpace(parsed.Choices[0].Message.Content), model, nil
}

func (c *answerClient) emitError(question, msg string) {
	out, _ := json.Marshal(map[string]any{
		"question": question,
		"answer":   "",
		"status":   "error",
		"error":    msg,
		"speak":    false,
	})
	c.hub.broadcast(out)
	fmt.Printf("voice-assist error: %s\n", msg)
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
