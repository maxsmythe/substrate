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

package glutton

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	gluttonpb "github.com/agent-substrate/substrate/internal/proto/glutton"
)

func writeDiskOp(key string, size int32) *gluttonpb.ScriptOp {
	return &gluttonpb.ScriptOp{Op: &gluttonpb.ScriptOp_WriteDisk{WriteDisk: &gluttonpb.WriteDiskRequest{Key: key, Size: size}}}
}

func readDiskOp(key string) *gluttonpb.ScriptOp {
	return &gluttonpb.ScriptOp{Op: &gluttonpb.ScriptOp_ReadDisk{ReadDisk: &gluttonpb.ReadDiskRequest{Key: key, ReadMode: gluttonpb.ReadMode_READ_MODE_DIGEST_ONLY}}}
}

func sleepOp(ms int64) *gluttonpb.ScriptOp {
	return &gluttonpb.ScriptOp{Op: &gluttonpb.ScriptOp_Sleep{Sleep: &gluttonpb.SleepRequest{DurationMs: ms}}}
}

// A script runs its ops in order, once, and each op has its usual effect:
// the file the first op wrote is there for the second to read.
func TestRunScriptRunsOpsInOrder(t *testing.T) {
	svc := newTestService(t)
	resp, err := svc.RunScript(context.Background(), &gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{
		writeDiskOp("scripted", 1024),
		readDiskOp("scripted"),
		{Op: &gluttonpb.ScriptOp_WriteRam{WriteRam: &gluttonpb.WriteRAMRequest{Key: "arr", Size: "8Ki"}}},
		{Op: &gluttonpb.ScriptOp_ReadRam{ReadRam: &gluttonpb.ReadRAMRequest{Key: "arr"}}},
		{Op: &gluttonpb.ScriptOp_BurnCpu{BurnCpu: &gluttonpb.BurnCPURequest{DurationMs: 1}}},
		sleepOp(1),
	}})
	if err != nil {
		t.Fatalf("RunScript: %v", err)
	}
	if resp.GetPasses() != 1 || resp.GetOpsRun() != 6 {
		t.Errorf("passes=%d ops_run=%d, want 1/6", resp.GetPasses(), resp.GetOpsRun())
	}
	if _, err := svc.ReadDisk(context.Background(), &gluttonpb.ReadDiskRequest{Key: "scripted"}); err != nil {
		t.Errorf("file written by the script is missing: %v", err)
	}
}

// With a loop budget the list repeats until the budget elapses, checked
// before each op, so the script ends within one op of the budget.
func TestRunScriptLoopsUntilTheBudget(t *testing.T) {
	svc := newTestService(t)
	start := time.Now()
	resp, err := svc.RunScript(context.Background(), &gluttonpb.RunScriptRequest{
		Ops:            []*gluttonpb.ScriptOp{sleepOp(10), sleepOp(10)},
		LoopDurationMs: 100,
	})
	if err != nil {
		t.Fatalf("RunScript: %v", err)
	}
	elapsed := time.Since(start)
	if elapsed < 100*time.Millisecond {
		t.Errorf("returned after %v, before the 100ms budget", elapsed)
	}
	if elapsed > 500*time.Millisecond {
		t.Errorf("returned after %v; the loop overran the budget by more than one op", elapsed)
	}
	if resp.GetPasses() < 2 || resp.GetOpsRun() < 4 {
		t.Errorf("passes=%d ops_run=%d, want at least 2/4 from a 100ms loop of 20ms passes", resp.GetPasses(), resp.GetOpsRun())
	}
	if resp.GetElapsedMs() < 100 {
		t.Errorf("elapsed_ms=%d, want >= 100", resp.GetElapsedMs())
	}
}

// Shape problems are rejected before any op runs; an op's own argument
// error keeps its code and names the op.
func TestRunScriptRejects(t *testing.T) {
	svc := newTestService(t)
	for _, tc := range []struct {
		name string
		req  *gluttonpb.RunScriptRequest
		code codes.Code
		msg  string
	}{
		{"empty", &gluttonpb.RunScriptRequest{}, codes.InvalidArgument, "ops is required"},
		{"unset op", &gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{{}}}, codes.InvalidArgument, "op 0: no op set"},
		{"negative sleep", &gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{sleepOp(-1)}}, codes.InvalidArgument, "op 0 (sleep)"},
		{"negative loop", &gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{sleepOp(0)}, LoopDurationMs: -1}, codes.InvalidArgument, "loop_duration_ms"},
		{"bad key keeps InvalidArgument", &gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{sleepOp(0), writeDiskOp("../x", 1)}}, codes.InvalidArgument, "op 1 (write_disk)"},
		{"missing file keeps NotFound", &gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{readDiskOp("nothing")}}, codes.NotFound, "op 0 (read_disk)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := svc.RunScript(context.Background(), tc.req)
			st, ok := status.FromError(err)
			if !ok || st.Code() != tc.code {
				t.Fatalf("error = %v, want code %v", err, tc.code)
			}
			if !strings.Contains(st.Message(), tc.msg) {
				t.Errorf("message %q does not mention %q", st.Message(), tc.msg)
			}
		})
	}
	if leftover := len(svc.ram); leftover != 0 {
		t.Errorf("rejected scripts left %d RAM arrays behind", leftover)
	}
}

// A shape error must stop the whole script before its first op: the valid
// first op of a list with a broken later op leaves no trace.
func TestRunScriptValidatesBeforeRunning(t *testing.T) {
	svc := newTestService(t)
	_, err := svc.RunScript(context.Background(), &gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{
		writeDiskOp("first", 16),
		{},
	}})
	if status.Code(err) != codes.InvalidArgument {
		t.Fatalf("error = %v, want InvalidArgument", err)
	}
	if _, err := svc.ReadDisk(context.Background(), &gluttonpb.ReadDiskRequest{Key: "first"}); status.Code(err) != codes.NotFound {
		t.Errorf("first op ran before the list was validated (ReadDisk err = %v)", err)
	}
}

// A caller that goes away ends the script at the next op boundary or
// mid-sleep, so a disconnected request does not keep the sandbox busy.
func TestRunScriptStopsWhenTheContextEnds(t *testing.T) {
	svc := newTestService(t)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := svc.RunScript(ctx, &gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{sleepOp(5000)}, LoopDurationMs: 10000})
	if status.Code(err) != codes.DeadlineExceeded {
		t.Fatalf("error = %v, want DeadlineExceeded", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("took %v to notice the context ended", elapsed)
	}
}

// The HTTP route carries the same request and maps the op's status code
// the way every other route does.
func TestRunScriptHTTPRoute(t *testing.T) {
	svc := newTestService(t)
	handler, err := Handler(ModeHTTP, svc)
	if err != nil {
		t.Fatal(err)
	}
	ts := httptest.NewServer(handler)
	defer ts.Close()

	post := func(req *gluttonpb.RunScriptRequest) (*http.Response, []byte) {
		t.Helper()
		body, _ := proto.Marshal(req)
		res, err := http.Post(ts.URL+RunScriptRoute, "application/x-protobuf", bytes.NewReader(body))
		if err != nil {
			t.Fatalf("POST %s: %v", RunScriptRoute, err)
		}
		defer res.Body.Close()
		out, _ := io.ReadAll(res.Body)
		return res, out
	}

	res, body := post(&gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{writeDiskOp("viahttp", 64), readDiskOp("viahttp")}})
	if res.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", res.StatusCode, body)
	}
	var resp gluttonpb.RunScriptResponse
	if err := proto.Unmarshal(body, &resp); err != nil {
		t.Fatal(err)
	}
	if resp.GetOpsRun() != 2 {
		t.Errorf("ops_run = %d, want 2", resp.GetOpsRun())
	}

	res, body = post(&gluttonpb.RunScriptRequest{Ops: []*gluttonpb.ScriptOp{readDiskOp("absent")}})
	if res.StatusCode != http.StatusNotFound || !strings.Contains(string(body), "op 0 (read_disk)") {
		t.Errorf("missing file over HTTP: status %d body %q, want 404 naming the op", res.StatusCode, body)
	}
}
