// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

//go:build !js

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

type faceEventLogger struct {
	mu       sync.Mutex
	path     string
	lastKey  string
	file     *os.File
}

func newFaceEventLogger() (*faceEventLogger, error) {
	dir := filepath.Join("facedetect")
	_ = os.MkdirAll(dir, 0o755)
	path := filepath.Join(dir, "events.jsonl")
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Face events log: %s\n", path)
	return &faceEventLogger{path: path, file: f}, nil
}

func (l *faceEventLogger) close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.file != nil {
		_ = l.file.Close()
	}
}

// logChange writes a line only when face presence/count changes.
func (l *faceEventLogger) logChange(raw []byte) {
	var payload struct {
		Present     bool    `json:"present"`
		Count       int     `json:"count"`
		UniqueCount int     `json:"unique_count"`
		WindowSec   int     `json:"window_sec"`
		Confidence  float64 `json:"confidence"`
		Status      string  `json:"status"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return
	}
	key := fmt.Sprintf("%v:%d:%d:%d", payload.Present, payload.Count, payload.UniqueCount, payload.WindowSec)
	l.mu.Lock()
	defer l.mu.Unlock()
	if key == l.lastKey {
		return
	}
	l.lastKey = key
	event := map[string]any{
		"ts":           time.Now().Format(time.RFC3339),
		"present":      payload.Present,
		"count":        payload.Count,
		"unique_count": payload.UniqueCount,
		"window_sec":   payload.WindowSec,
		"confidence":   payload.Confidence,
		"status":       payload.Status,
	}
	b, err := json.Marshal(event)
	if err != nil {
		return
	}
	_, _ = l.file.Write(append(b, '\n'))
	_ = l.file.Sync()
	fmt.Printf("face-event: %s\n", b)
}
