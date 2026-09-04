//go:build linux

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

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/agent-substrate/substrate/internal/ateompath"
	"github.com/agent-substrate/substrate/internal/resources"
)

func TestKillArgs(t *testing.T) {
	r := &runsc{
		path:     "/usr/bin/runsc",
		actorUID: "test-actor-123",
	}

	got := r.killArgs("my-container", "SIGTERM")
	want := []string{
		"-log-format", "json",
		"--alsologtostderr",
		"-root", ateompath.RunSCStateDir("test-actor-123"),
		"kill",
		"my-container",
		"SIGTERM",
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("killArgs() = %v, want %v", got, want)
	}
}

func TestWaitArgs(t *testing.T) {
	r := &runsc{
		path:     "/usr/bin/runsc",
		actorUID: "test-actor-123",
	}

	got := r.waitArgs("my-container")
	want := []string{
		"-log-format", "json",
		"--alsologtostderr",
		"-root", ateompath.RunSCStateDir("test-actor-123"),
		"wait",
		"my-container",
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("waitArgs() = %v, want %v", got, want)
	}
}

func TestPauseArgs(t *testing.T) {
	r := &runsc{
		path:     "/usr/bin/runsc",
		actorUID: "test-actor-123",
	}

	got := r.pauseArgs("pause")
	want := []string{
		"-log-format", "json",
		"--alsologtostderr",
		"-root", ateompath.RunSCStateDir("test-actor-123"),
		"pause",
		"pause",
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("pauseArgs() = %v, want %v", got, want)
	}
}

func TestResumeArgs(t *testing.T) {
	r := &runsc{
		path:     "/usr/bin/runsc",
		actorUID: "test-actor-123",
	}

	got := r.resumeArgs("pause")
	want := []string{
		"-log-format", "json",
		"--alsologtostderr",
		"-root", ateompath.RunSCStateDir("test-actor-123"),
		"resume",
		"pause",
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("resumeArgs() = %v, want %v", got, want)
	}
}

func TestRestoreArgs(t *testing.T) {
	r := &runsc{
		path:     "/usr/bin/runsc",
		actorUID: "test-actor-123",
	}

	got := r.restoreArgs("pause", "/var/lib/ate/restore-state")
	want := []string{
		"-log-format", "json",
		"--alsologtostderr",
		"-root", ateompath.RunSCStateDir("test-actor-123"),
		"--cpu-num-from-quota",
		"restore",
		"-bundle", ateompath.OCIBundlePath("test-actor-123", "pause"),
		"-image-path", "/var/lib/ate/restore-state",
		"-pid-file", ateompath.PIDFilePath("test-actor-123", "pause"),
		"-background",
		"-direct",
		"-detach",
		"pause",
	}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("restoreArgs() = %v, want %v", got, want)
	}
}

func TestRestoreFault(t *testing.T) {
	tests := []struct {
		name     string
		atespace string
		want     bool
	}{
		{name: "golden actor is exempt", atespace: resources.GoldenActorAtespace, want: false},
		{name: "regular actor faults", atespace: "team-a", want: true},
		{name: "empty atespace faults", atespace: "", want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &runsc{actorUID: "test-actor-123", atespace: tt.atespace}
			if got := r.restoreFault(); got != tt.want {
				t.Errorf("restoreFault() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFakeRestoreFailure(t *testing.T) {
	r := &runsc{actorUID: "test-actor-123", atespace: "team-a"}
	var out bytes.Buffer

	err := r.fakeRestoreFailure(context.Background(), &out, "pause")
	if !errors.Is(err, errRestoreFault) {
		t.Fatalf("fakeRestoreFailure() error = %v, want errRestoreFault", err)
	}
	if want := "while running `runsc restore`: exit status 128"; err.Error() != want {
		t.Errorf("fakeRestoreFailure() error = %q, want %q", err.Error(), want)
	}

	var line map[string]string
	if jerr := json.Unmarshal(bytes.TrimSpace(out.Bytes()), &line); jerr != nil {
		t.Fatalf("log output %q is not a JSON object: %v", out.String(), jerr)
	}
	if line["level"] != "ERROR" {
		t.Errorf("log level = %q, want ERROR", line["level"])
	}
	if !strings.Contains(line["msg"], `"pause"`) {
		t.Errorf("log msg %q does not name the container", line["msg"])
	}
}
