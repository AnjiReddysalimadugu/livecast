// SPDX-FileCopyrightText: 2026 The Pion community <https://pion.ly>
// SPDX-License-Identifier: MIT

//go:build !js

package main

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/pion/webrtc/v4"
)

func TestNormalizeRoomID(t *testing.T) {
	cases := []struct {
		in      string
		want    string
		wantErr bool
	}{
		{"abc", "abc", false},
		{"  AbC123  ", "abc123", false},
		{"room_1-2", "room_1-2", false},
		{"ab", "", true},                  // too short
		{"", "", true},                    // empty
		{"   ", "", true},                 // blank
		{strings.Repeat("a", 25), "", true}, // too long
		{"bad room", "", true},            // space
		{"bad@id", "", true},              // illegal char
		{"OK_ROOM", "ok_room", false},
	}
	for _, tc := range cases {
		got, err := normalizeRoomID(tc.in)
		if tc.wantErr {
			if err == nil {
				t.Fatalf("normalizeRoomID(%q) expected error, got %q", tc.in, got)
			}
			continue
		}
		if err != nil {
			t.Fatalf("normalizeRoomID(%q) unexpected err: %v", tc.in, err)
		}
		if got != tc.want {
			t.Fatalf("normalizeRoomID(%q)=%q want %q", tc.in, got, tc.want)
		}
	}
}

func TestOpenRelayStaticCreds(t *testing.T) {
	u, c := openRelayStaticCreds("livecast", time.Hour)
	if !strings.Contains(u, ":livecast") {
		t.Fatalf("username shape: %q", u)
	}
	if c == "" || len(c) < 10 {
		t.Fatalf("credential empty/short: %q", c)
	}
	// Fresh creds should differ by timestamp second boundaries occasionally;
	// at least encoding must be stable for same username.
	u2, c2 := openRelayStaticCreds("livecast", time.Hour)
	if !strings.HasSuffix(u2, ":livecast") || c2 == "" {
		t.Fatalf("second creds invalid")
	}
}

func TestLoadICEServersHasTURN(t *testing.T) {
	t.Setenv("LIVECAST_CLOUD", "")
	t.Setenv("RENDER", "")
	iceCache.mu.Lock()
	iceCache.servers = nil
	iceCache.source = ""
	iceCache.at = time.Time{}
	iceCache.mu.Unlock()

	servers := loadICEServers()
	if len(servers) < 1 {
		t.Fatalf("expected ICE servers, got %d", len(servers))
	}
	src := iceSource()
	if src == "stun-only" {
		t.Skip("cloud stun-only mode — no TURN expected")
	}
	hasTURN := false
	for _, s := range servers {
		for _, u := range s.URLs {
			if strings.HasPrefix(u, "turn:") || strings.HasPrefix(u, "turns:") {
				hasTURN = true
				if s.Username == "" || s.Credential == nil || s.Credential == "" {
					t.Fatalf("TURN missing creds for %v", s.URLs)
				}
			}
		}
	}
	if !hasTURN {
		t.Fatal("no TURN urls in ICE config")
	}
}

func TestCloudWithoutTURNIsStunOnly(t *testing.T) {
	t.Setenv("LIVECAST_CLOUD", "1")
	t.Setenv("RENDER", "")
	t.Setenv("METERED_DOMAIN", "")
	t.Setenv("METERED_API_KEY", "")
	t.Setenv("TURN_URLS", "")
	t.Setenv("TURN_USERNAME", "")
	t.Setenv("TURN_CREDENTIAL", "")
	iceCache.mu.Lock()
	iceCache.servers = nil
	iceCache.source = ""
	iceCache.at = time.Time{}
	iceCache.mu.Unlock()

	_ = loadICEServers()
	if iceSource() != "stun-only" {
		t.Fatalf("want stun-only, got %s", iceSource())
	}
	if turnReady() {
		t.Fatal("turn should not be ready without Metered/TURN env")
	}
}

func TestICEServersJSONValid(t *testing.T) {
	raw := iceServersJSON()
	var arr []map[string]any
	if err := json.Unmarshal(raw, &arr); err != nil {
		t.Fatalf("json: %v body=%s", err, raw)
	}
	if len(arr) == 0 {
		t.Fatal("empty ice json")
	}
	for _, item := range arr {
		if _, ok := item["urls"]; !ok {
			t.Fatalf("missing urls: %#v", item)
		}
		if cred, ok := item["credential"]; ok {
			if s, isStr := cred.(string); isStr && (s == "<nil>" || s == "nil") {
				t.Fatalf("bad credential serialization: %#v", item)
			}
		}
	}
}

func TestWithFreshICE(t *testing.T) {
	cfg := withFreshICE(webrtcConfigStub())
	if len(cfg.ICEServers) == 0 {
		t.Fatal("fresh ICE empty")
	}
}

func webrtcConfigStub() webrtc.Configuration {
	return webrtc.Configuration{}
}
