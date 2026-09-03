// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cachesync

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// captureLogs routes the default slog logger into a buffer for the test.
func captureLogs(t *testing.T) *bytes.Buffer {
	t.Helper()
	var buf bytes.Buffer
	prev := slog.Default()
	slog.SetDefault(slog.New(slog.NewTextHandler(&buf, nil)))
	t.Cleanup(func() { slog.SetDefault(prev) })
	return &buf
}

func TestWaitReturnsQuietlyWhenSyncIsFast(t *testing.T) {
	logs := captureLogs(t)
	synced := func() bool { return true }

	if !wait(context.Background(), "fast", time.Hour, synced) {
		t.Fatal("wait = false for an already-synced informer")
	}
	if logs.Len() != 0 {
		t.Fatalf("fast sync produced logs, want none:\n%s", logs)
	}
}

func TestWaitReportsSlowSyncThenCompletion(t *testing.T) {
	logs := captureLogs(t)
	var ready atomic.Bool
	synced := ready.Load

	done := make(chan bool, 1)
	go func() { done <- wait(context.Background(), "slow", 10*time.Millisecond, synced) }()

	// Let at least one warning fire before the cache syncs.
	time.Sleep(50 * time.Millisecond)
	ready.Store(true)

	select {
	case ok := <-done:
		if !ok {
			t.Fatal("wait = false, want true after the informer synced")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("wait did not return after the informer synced")
	}

	out := logs.String()
	if !strings.Contains(out, "informer cache not yet synced") || !strings.Contains(out, "informer=slow") {
		t.Errorf("missing stall warning in logs:\n%s", out)
	}
	if !strings.Contains(out, "informer cache sync finished") || !strings.Contains(out, "synced=true") {
		t.Errorf("missing completion log after a slow sync:\n%s", out)
	}
}

func TestWaitReturnsFalseWhenContextEnds(t *testing.T) {
	logs := captureLogs(t)
	never := func() bool { return false }
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan bool, 1)
	go func() { done <- wait(ctx, "cancelled", 10*time.Millisecond, never) }()
	time.Sleep(30 * time.Millisecond)
	cancel()

	select {
	case ok := <-done:
		if ok {
			t.Fatal("wait = true after context cancellation, want false")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("wait did not return after context cancellation")
	}
	if !strings.Contains(logs.String(), "synced=false") {
		t.Errorf("completion log should record synced=false:\n%s", logs)
	}
}
