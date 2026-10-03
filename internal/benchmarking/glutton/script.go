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
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	gluttonpb "github.com/agent-substrate/substrate/internal/proto/glutton"
)

// maxScriptOps bounds one request's op list. A script that needs more ops
// than this wants the loop, not a longer list.
const maxScriptOps = 4096

// RunScript runs the request's ops in order, in-process, and holds the
// request open until they finish. With loop_duration_ms set the list is
// run pass after pass until the budget elapses, checked before each op.
// The list is checked for shape before anything runs; an op's own argument
// errors surface when it runs, with the op's index and kind prepended and
// its status code kept, so a caller can tell a bad script from a failed op.
func (s *Service) RunScript(ctx context.Context, req *gluttonpb.RunScriptRequest) (*gluttonpb.RunScriptResponse, error) {
	if err := validateScript(req); err != nil {
		return nil, err
	}
	start := time.Now()
	var deadline time.Time
	if req.GetLoopDurationMs() > 0 {
		deadline = start.Add(time.Duration(req.GetLoopDurationMs()) * time.Millisecond)
	}
	resp := &gluttonpb.RunScriptResponse{}
	for {
		resp.Passes++
		for i, op := range req.GetOps() {
			if !deadline.IsZero() && !time.Now().Before(deadline) {
				resp.ElapsedMs = time.Since(start).Milliseconds()
				return resp, nil
			}
			if err := ctx.Err(); err != nil {
				return nil, status.FromContextError(err).Err()
			}
			if err := s.runOp(ctx, op); err != nil {
				return nil, prefixOpError(i, op, err)
			}
			resp.OpsRun++
		}
		if deadline.IsZero() {
			resp.ElapsedMs = time.Since(start).Milliseconds()
			return resp, nil
		}
	}
}

// validateScript rejects what can be seen without running anything: an
// empty or oversized list, an op with nothing set, a negative sleep or
// loop budget.
func validateScript(req *gluttonpb.RunScriptRequest) error {
	if len(req.GetOps()) == 0 {
		return status.Error(codes.InvalidArgument, "ops is required")
	}
	if len(req.GetOps()) > maxScriptOps {
		return status.Errorf(codes.InvalidArgument, "ops has %d entries, max %d", len(req.GetOps()), maxScriptOps)
	}
	if req.GetLoopDurationMs() < 0 {
		return status.Error(codes.InvalidArgument, "loop_duration_ms must be non-negative")
	}
	for i, op := range req.GetOps() {
		switch typed := op.GetOp().(type) {
		case nil:
			return status.Errorf(codes.InvalidArgument, "op %d: no op set", i)
		case *gluttonpb.ScriptOp_Sleep:
			if typed.Sleep.GetDurationMs() < 0 {
				return status.Errorf(codes.InvalidArgument, "op %d (sleep): duration_ms must be non-negative", i)
			}
		}
	}
	return nil
}

// runOp dispatches one op to the service method it names. Results other
// than errors are dropped: a ReadDisk's bytes or a BurnCPU's iteration
// count would only be shipped back to a caller who asked for timings.
func (s *Service) runOp(ctx context.Context, op *gluttonpb.ScriptOp) error {
	var err error
	switch typed := op.GetOp().(type) {
	case *gluttonpb.ScriptOp_WriteRam:
		_, err = s.WriteRAM(ctx, typed.WriteRam)
	case *gluttonpb.ScriptOp_ReadRam:
		_, err = s.ReadRAM(ctx, typed.ReadRam)
	case *gluttonpb.ScriptOp_WriteDisk:
		_, err = s.WriteDisk(ctx, typed.WriteDisk)
	case *gluttonpb.ScriptOp_ReadDisk:
		_, err = s.ReadDisk(ctx, typed.ReadDisk)
	case *gluttonpb.ScriptOp_BurnCpu:
		_, err = s.BurnCPU(ctx, typed.BurnCpu)
	case *gluttonpb.ScriptOp_UseCpu:
		_, err = s.UseCPU(ctx, typed.UseCpu)
	case *gluttonpb.ScriptOp_Ingest:
		_, err = s.Ingest(ctx, typed.Ingest)
	case *gluttonpb.ScriptOp_OpenFd:
		_, err = s.OpenFD(ctx, typed.OpenFd)
	case *gluttonpb.ScriptOp_Sleep:
		err = sleep(ctx, time.Duration(typed.Sleep.GetDurationMs())*time.Millisecond)
	default:
		err = status.Errorf(codes.InvalidArgument, "unknown op %T", typed)
	}
	return err
}

// sleep idles for duration or until ctx ends, whichever comes first.
func sleep(ctx context.Context, duration time.Duration) error {
	if duration <= 0 {
		return nil
	}
	timer := time.NewTimer(duration)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return status.FromContextError(ctx.Err()).Err()
	}
}

// opKindName is the proto field name of the op that is set, for messages.
func opKindName(op *gluttonpb.ScriptOp) string {
	switch op.GetOp().(type) {
	case *gluttonpb.ScriptOp_WriteRam:
		return "write_ram"
	case *gluttonpb.ScriptOp_ReadRam:
		return "read_ram"
	case *gluttonpb.ScriptOp_WriteDisk:
		return "write_disk"
	case *gluttonpb.ScriptOp_ReadDisk:
		return "read_disk"
	case *gluttonpb.ScriptOp_BurnCpu:
		return "burn_cpu"
	case *gluttonpb.ScriptOp_UseCpu:
		return "use_cpu"
	case *gluttonpb.ScriptOp_Ingest:
		return "ingest"
	case *gluttonpb.ScriptOp_OpenFd:
		return "open_fd"
	case *gluttonpb.ScriptOp_Sleep:
		return "sleep"
	}
	return "unset"
}

// prefixOpError names the failing op in err's message, keeping its status
// code so the HTTP mapping and the caller's classification still apply.
func prefixOpError(i int, op *gluttonpb.ScriptOp, err error) error {
	st, ok := status.FromError(err)
	if !ok {
		return status.Errorf(codes.Internal, "op %d (%s): %v", i, opKindName(op), err)
	}
	return status.Error(st.Code(), fmt.Sprintf("op %d (%s): %s", i, opKindName(op), st.Message()))
}
