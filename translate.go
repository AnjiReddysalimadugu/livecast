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
	"path/filepath"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/pion/webrtc/v4"
)

type translateHub struct {
	mu    sync.Mutex
	last  []byte
	chans []*webrtc.DataChannel
}

func (h *translateHub) snapshot() []byte {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.last) == 0 {
		return []byte(`{"source":"","translated":"","target":"en","status":"waiting"}`)
	}
	return append([]byte(nil), h.last...)
}

func (h *translateHub) reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.chans = nil
}

func (h *translateHub) add(dc *webrtc.DataChannel) {
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

func (h *translateHub) broadcast(msg []byte) {
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

type translateConfig struct {
	TargetLang string `json:"target_lang"` // en, te, hi, ta, ...
	Enabled    bool   `json:"enabled"`
}

func translateConfigPath() string {
	return filepath.Join("translate", "config.json")
}

func loadTranslateConfig() translateConfig {
	cfg := translateConfig{TargetLang: "en", Enabled: true}
	b, err := os.ReadFile(translateConfigPath())
	if err != nil {
		return cfg
	}
	_ = json.Unmarshal(b, &cfg)
	if strings.TrimSpace(cfg.TargetLang) == "" {
		cfg.TargetLang = "en"
	}
	return cfg
}

func saveTranslateConfig(cfg translateConfig) error {
	if err := os.MkdirAll("translate", 0o755); err != nil {
		return err
	}
	if strings.TrimSpace(cfg.TargetLang) == "" {
		cfg.TargetLang = "en"
	}
	b, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(translateConfigPath(), b, 0o644)
}

type liveTranslator struct {
	hub    *translateHub
	client *http.Client
	ollama string
	model  string

	mu       sync.Mutex
	lastSrc  string
	inflight bool
}

func newLiveTranslator(hub *translateHub) *liveTranslator {
	ollama := strings.TrimSpace(os.Getenv("OLLAMA_URL"))
	if ollama == "" {
		ollama = "http://127.0.0.1:11434"
	}
	model := strings.TrimSpace(os.Getenv("OLLAMA_MODEL"))
	if model == "" {
		model = "llama3.2:3b"
	}
	return &liveTranslator{
		hub:    hub,
		ollama: strings.TrimRight(ollama, "/"),
		model:  model,
		client: &http.Client{Timeout: 60 * time.Second},
	}
}

func langName(code string) string {
	switch strings.ToLower(code) {
	case "en":
		return "English"
	case "te":
		return "Telugu"
	case "hi":
		return "Hindi"
	case "ta":
		return "Tamil"
	case "kn":
		return "Kannada"
	case "ml":
		return "Malayalam"
	case "es":
		return "Spanish"
	case "fr":
		return "French"
	case "de":
		return "German"
	default:
		return code
	}
}

func hasLetters(s string) bool {
	n := 0
	for _, r := range s {
		if unicode.IsLetter(r) {
			n++
			if n >= 3 {
				return true
			}
		}
	}
	return false
}

func hasIndicScript(s string) bool {
	for _, r := range s {
		if (r >= 0x0900 && r <= 0x097F) || // Devanagari
			(r >= 0x0B80 && r <= 0x0BFF) || // Tamil
			(r >= 0x0C00 && r <= 0x0C7F) || // Telugu
			(r >= 0x0C80 && r <= 0x0CFF) || // Kannada
			(r >= 0x0D00 && r <= 0x0D7F) { // Malayalam
			return true
		}
	}
	return false
}

func mostlyTargetAlready(source, targetCode string) bool {
	src := strings.TrimSpace(source)
	if src == "" {
		return false
	}
	switch strings.ToLower(targetCode) {
	case "en":
		// Already English-looking: Latin letters, no Indic script.
		return !hasIndicScript(src) && hasLetters(src)
	case "te":
		for _, r := range src {
			if r >= 0x0C00 && r <= 0x0C7F {
				return true
			}
		}
		return false
	case "hi":
		for _, r := range src {
			if r >= 0x0900 && r <= 0x097F {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// newestUtterance keeps the latest clause so overlapping Whisper chunks
// do not keep re-translating the whole buffer.
func newestUtterance(text string) string {
	t := strings.TrimSpace(text)
	if t == "" {
		return t
	}
	// Prefer text after the last sentence end.
	for _, sep := range []string{". ", "? ", "! ", "。", "\n"} {
		if i := strings.LastIndex(t, sep); i >= 0 && i+len(sep) < len(t) {
			tail := strings.TrimSpace(t[i+len(sep):])
			if hasLetters(tail) {
				t = tail
			}
		}
	}
	words := strings.Fields(t)
	if len(words) > 28 {
		words = words[len(words)-28:]
		t = strings.Join(words, " ")
	}
	return t
}

func (t *liveTranslator) onTranscriptJSON(raw []byte) {
	if t == nil || t.hub == nil {
		return
	}
	cfg := loadTranslateConfig()
	if !cfg.Enabled {
		return
	}
	var payload struct {
		Text   string `json:"text"`
		Status string `json:"status"`
		Lang   string `json:"lang"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return
	}
	src := newestUtterance(payload.Text)
	if payload.Status != "ok" || !hasLetters(src) {
		return
	}

	t.mu.Lock()
	if src == t.lastSrc {
		t.mu.Unlock()
		return
	}
	// If busy, replace pending source so we always translate the latest line.
	if t.inflight {
		t.lastSrc = src
		t.mu.Unlock()
		return
	}
	t.lastSrc = src
	t.inflight = true
	t.mu.Unlock()

	go t.runTranslate(src, cfg.TargetLang, payload.Lang)
}

func (t *liveTranslator) runTranslate(source, target, detected string) {
	defer func() {
		t.mu.Lock()
		pending := t.lastSrc
		t.inflight = false
		t.mu.Unlock()
		// Catch up if a newer utterance arrived while we were busy.
		if pending != "" && pending != source {
			t.mu.Lock()
			if !t.inflight {
				t.inflight = true
				t.mu.Unlock()
				go t.runTranslate(pending, target, detected)
				return
			}
			t.mu.Unlock()
		}
	}()

	if mostlyTargetAlready(source, target) {
		msg, _ := json.Marshal(map[string]any{
			"source":     source,
			"translated": source,
			"target":     target,
			"detected":   detected,
			"status":     "same",
		})
		t.hub.broadcast(msg)
		return
	}

	pending, _ := json.Marshal(map[string]any{
		"source":     source,
		"translated": "",
		"target":     target,
		"detected":   detected,
		"status":     "translating",
	})
	t.hub.broadcast(pending)

	out, err := t.translate(source, target, detected)
	if err != nil {
		msg, _ := json.Marshal(map[string]any{
			"source":     source,
			"translated": "",
			"target":     target,
			"status":     "error",
			"error":      err.Error(),
		})
		t.hub.broadcast(msg)
		fmt.Printf("translate error: %v\n", err)
		return
	}
	msg, _ := json.Marshal(map[string]any{
		"source":     source,
		"translated": out,
		"target":     target,
		"detected":   detected,
		"status":     "ok",
	})
	t.hub.broadcast(msg)
	fmt.Printf("translate: %q → %q\n", source, out)
}

func (t *liveTranslator) translate(source, targetCode, detected string) (string, error) {
	target := langName(targetCode)
	srcHint := detected
	if srcHint == "" {
		srcHint = "auto"
	}
	prompt := fmt.Sprintf(
		"Translate this live meeting caption into %s.\n"+
			"Source language hint: %s.\n"+
			"Rules:\n"+
			"- Output ONLY the %s translation.\n"+
			"- Keep names and numbers.\n"+
			"- Do not explain, quote, or add extra words.\n"+
			"- If already in %s, return it unchanged.\n\n"+
			"Text:\n%s",
		target, srcHint, target, target, source,
	)
	body, _ := json.Marshal(map[string]any{
		"model":  t.model,
		"stream": false,
		"options": map[string]any{
			"temperature": 0.1,
			"num_predict": 180,
		},
		"messages": []map[string]string{
			{
				"role": "system",
				"content": "You are a precise live translator for meetings (like Microsoft Teams captions). " +
					"Never answer questions. Never summarize. Only translate.",
			},
			{"role": "user", "content": prompt},
		},
	})
	req, err := http.NewRequest(http.MethodPost, t.ollama+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := t.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama unreachable (is it running?): %w", err)
	}
	defer res.Body.Close()
	raw, _ := io.ReadAll(res.Body)
	if res.StatusCode >= 300 {
		return "", fmt.Errorf("ollama HTTP %d: %s", res.StatusCode, string(raw))
	}
	var parsed struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return "", err
	}
	out := strings.TrimSpace(parsed.Message.Content)
	out = strings.Trim(out, "\"'")
	// Strip common model preambles.
	for _, p := range []string{"Translation:", "Translated:", "Here is the translation:", "English:", "Telugu:"} {
		if strings.HasPrefix(strings.ToLower(out), strings.ToLower(p)) {
			out = strings.TrimSpace(out[len(p):])
		}
	}
	if out == "" {
		return "", fmt.Errorf("empty translation")
	}
	return out, nil
}

func createTranslateChannel(pc *webrtc.PeerConnection, hub *translateHub) error {
	if hub == nil {
		return nil
	}
	id := uint16(4)
	negotiated := true
	dc, err := pc.CreateDataChannel("translations", &webrtc.DataChannelInit{
		ID:         &id,
		Negotiated: &negotiated,
	})
	if err != nil {
		return err
	}
	hub.add(dc)
	return nil
}
