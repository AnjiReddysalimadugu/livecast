package main

import (
	"crypto/hmac"
	"crypto/sha1"
	"encoding/base64"
	"fmt"
	"net"
	"os"
	"time"

	"github.com/pion/logging"
	"github.com/pion/turn/v4"
)

func creds() (string, string) {
	secret := "openrelayprojectsecret"
	expire := time.Now().Add(time.Hour).Unix()
	username := fmt.Sprintf("%d:livecast", expire)
	mac := hmac.New(sha1.New, []byte(secret))
	_, _ = mac.Write([]byte(username))
	return username, base64.StdEncoding.EncodeToString(mac.Sum(nil))
}

func tryHost(host string, port int, useTLS bool) error {
	user, pass := creds()
	fmt.Printf("trying %s:%d user=%s\n", host, port, user)

	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", host, port), 8*time.Second)
	if err != nil {
		return fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()

	cfg := &turn.ClientConfig{
		STUNServerAddr: fmt.Sprintf("%s:%d", host, port),
		TURNServerAddr: fmt.Sprintf("%s:%d", host, port),
		Username:       user,
		Password:       pass,
		Conn:           turn.NewSTUNConn(conn),
		LoggerFactory:  logging.NewDefaultLoggerFactory(),
	}
	client, err := turn.NewClient(cfg)
	if err != nil {
		return fmt.Errorf("client: %w", err)
	}
	defer client.Close()
	if err := client.Listen(); err != nil {
		return fmt.Errorf("listen: %w", err)
	}
	relay, err := client.Allocate()
	if err != nil {
		return fmt.Errorf("allocate: %w", err)
	}
	fmt.Printf("OK relay=%v\n", relay.LocalAddr())
	_ = relay.Close()
	return nil
}

func main() {
	hosts := []struct {
		h string
		p int
	}{
		{"staticauth.openrelay.metered.ca", 80},
		{"staticauth.openrelay.metered.ca", 443},
		{"openrelay.metered.ca", 80},
		{"global.relay.metered.ca", 80},
	}
	for _, h := range hosts {
		if err := tryHost(h.h, h.p, false); err != nil {
			fmt.Printf("FAIL %s:%d → %v\n", h.h, h.p, err)
		}
	}
	os.Exit(0)
}
