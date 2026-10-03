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

package agentwalk

import (
	"context"
	"math"
	"math/rand/v2"
	"net/http"
	"slices"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/agent-substrate/substrate/internal/benchmarking/boomer/dynconfig"
	"github.com/agent-substrate/substrate/internal/benchmarking/boomer/userclass"
	"github.com/agent-substrate/substrate/internal/benchmarking/glutton"
	"github.com/agent-substrate/substrate/internal/benchmarking/glutton/fake"
	gluttonpb "github.com/agent-substrate/substrate/internal/proto/glutton"
	"github.com/agent-substrate/substrate/pkg/proto/ateapipb"
	"go.opentelemetry.io/otel"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func testRand() *rand.Rand { return rand.New(rand.NewPCG(1, 2)) }

// testKnobs starts from the section's defaults and applies edit, the way a
// run starts from the master's flag defaults and the operator changes a few.
func testKnobs(edit func(k *walkKnobs)) walkKnobs {
	k := walkCodec.Defaults
	if edit != nil {
		edit(&k)
	}
	return k
}

// onlyWeights sets all four action weights at once, so a test can zero the
// actions it does not want and make the walk deterministic.
func onlyWeights(llm, short, long, done float64) func(k *walkKnobs) {
	return func(k *walkKnobs) {
		k.WeightLLM, k.WeightShort, k.WeightLong, k.WeightDone = llm, short, long, done
	}
}

// The section's defaults resolve to the documented values, and set knobs
// carry through in the walk's units.
func TestResolveDefaults(t *testing.T) {
	p := resolve(walkCodec.Defaults)
	want := params{
		cronInterval: 30 * time.Minute,
		idleCPU:      0.25,
		weights:      [numActions]float64{50, 30, 10, 10},
		thinkMu:      2.0414,
		thinkSigma:   0.8674,
		shortMean:    6 * time.Second,
		longMean:     time.Minute,
		ramSize:      32 << 20,
		diskSize:     16 << 20,
		maxActions:   50,
		template:     "glutton",
	}
	if p != want {
		t.Errorf("resolve(defaults) =\n %+v, want\n %+v", p, want)
	}

	p = resolve(testKnobs(func(k *walkKnobs) {
		k.CronInterval = dynconfig.Seconds(5 * time.Minute)
		k.IdleCPU = 0.1
		k.RAMSize = "1Gi"
		k.DiskSize = "4Ki"
		k.Template = "glutton-big"
	}))
	if p.cronInterval != 5*time.Minute || p.idleCPU != 0.1 || p.ramSize != 1<<30 || p.diskSize != 4096 || p.template != "glutton-big" {
		t.Errorf("set knobs did not carry: %+v", p)
	}
	// A holder built without validation can carry a zero interval; the
	// tick arithmetic must not divide by it.
	if p := resolve(testKnobs(func(k *walkKnobs) { k.CronInterval = 0 })); p.cronInterval != 30*time.Minute {
		t.Errorf("zero interval resolved to %v, want the default", p.cronInterval)
	}
}

// The section refuses the values the walk cannot run on, so a bad flag
// from the master is kept out of the holder.
func TestValidateKnobs(t *testing.T) {
	if err := validateKnobs(walkCodec.Defaults); err != nil {
		t.Fatalf("defaults do not validate: %v", err)
	}
	for name, edit := range map[string]func(k *walkKnobs){
		"zero cron interval": func(k *walkKnobs) { k.CronInterval = 0 },
		"idle cpu above one": func(k *walkKnobs) { k.IdleCPU = 1.5 },
		"negative weight":    func(k *walkKnobs) { k.WeightLong = -1 },
		"all weights zero":   onlyWeights(0, 0, 0, 0),
		"zero think sigma":   func(k *walkKnobs) { k.ThinkSigma = 0 },
		"zero short seconds": func(k *walkKnobs) { k.ShortSeconds = 0 },
		"negative long":      func(k *walkKnobs) { k.LongSeconds = -1 },
		"unparseable ram":    func(k *walkKnobs) { k.RAMSize = "lots" },
		"zero disk":          func(k *walkKnobs) { k.DiskSize = "0" },
		"zero max actions":   func(k *walkKnobs) { k.MaxActions = 0 },
		"empty template":     func(k *walkKnobs) { k.Template = "" },
		"bad resume mode":    func(k *walkKnobs) { k.ResumeMode = "sometimes" },
	} {
		t.Run(name, func(t *testing.T) {
			if err := validateKnobs(testKnobs(edit)); err == nil {
				t.Error("validateKnobs accepted the knobs")
			}
		})
	}
}

// The walk's rules: done is never first, never before an LLM query, and
// the cap winds a session up with an LLM query first if it owes one.
func TestPickConstraints(t *testing.T) {
	r := testRand()
	p := resolve(testKnobs(onlyWeights(1, 1, 1, 1000)))
	for i := 0; i < 1000; i++ {
		if a := p.pick(r, walkState{}); a == actionDone {
			t.Fatal("done chosen as the first action")
		}
		if a := p.pick(r, walkState{actions: 5}); a == actionDone {
			t.Fatal("done chosen before any LLM query")
		}
	}
	if a := p.pick(r, walkState{actions: 1, llmCalled: true}); a != actionDone {
		t.Errorf("with done at weight 1000 and eligible, pick = %v", a)
	}
	if a := p.pick(r, walkState{actions: p.maxActions}); a != actionLLM {
		t.Errorf("at the cap without an LLM query, pick = %v, want llm", a)
	}
	if a := p.pick(r, walkState{actions: p.maxActions + 1, llmCalled: true}); a != actionDone {
		t.Errorf("at the cap after an LLM query, pick = %v, want done", a)
	}
	// Only done has weight, and done is not allowed yet: the session ends
	// since nothing else may happen.
	onlyDone := resolve(testKnobs(onlyWeights(0, 0, 0, 1)))
	if a := onlyDone.pick(r, walkState{}); a != actionDone {
		t.Errorf("with nothing eligible, pick = %v, want done", a)
	}
}

// The draw follows the weights: a zero weight never comes up, and the
// shares of a long run land near the configured proportions.
func TestPickFollowsWeights(t *testing.T) {
	r := testRand()
	p := resolve(testKnobs(onlyWeights(60, 40, 0, 0)))
	var counts [numActions]int
	const n = 20000
	for i := 0; i < n; i++ {
		counts[p.pick(r, walkState{actions: 3, llmCalled: true})]++
	}
	if counts[actionLong] != 0 || counts[actionDone] != 0 {
		t.Errorf("zero-weight actions were drawn: %v", counts)
	}
	if share := float64(counts[actionLLM]) / n; math.Abs(share-0.6) > 0.02 {
		t.Errorf("llm share = %.3f, want ~0.6", share)
	}
}

// The think time is log-normal with SWE-perf's moments: median near 7.7s,
// 90th percentile near 23.4s.
func TestThinkDistribution(t *testing.T) {
	r := testRand()
	p := resolve(walkCodec.Defaults)
	const n = 20000
	samples := make([]float64, n)
	for i := range samples {
		samples[i] = p.think(r).Seconds()
	}
	sort.Float64s(samples)
	if med := samples[n/2]; math.Abs(med-7.7) > 0.4 {
		t.Errorf("median think = %.2fs, want ~7.7s", med)
	}
	if p90 := samples[n*9/10]; math.Abs(p90-23.4) > 1.5 {
		t.Errorf("p90 think = %.2fs, want ~23.4s", p90)
	}
}

// Operation lengths stay within the spread around their means.
func TestDurationSpread(t *testing.T) {
	r := testRand()
	p := resolve(testKnobs(func(k *walkKnobs) {
		k.ShortSeconds = dynconfig.Seconds(10 * time.Second)
		k.LongSeconds = dynconfig.Seconds(100 * time.Second)
	}))
	for i := 0; i < 1000; i++ {
		if d := p.shortDuration(r); d < 5*time.Second || d > 15*time.Second {
			t.Fatalf("short duration %v outside [5s, 15s]", d)
		}
		if d := p.longDuration(r); d < 50*time.Second || d > 150*time.Second {
			t.Fatalf("long duration %v outside [50s, 150s]", d)
		}
	}
}

// Each script is the request RunScript expects for its activity: a single
// burn for CPU, a paced write/read loop bounded by the duration for disk
// and RAM, sized by the knobs.
func TestScripts(t *testing.T) {
	p := resolve(testKnobs(func(k *walkKnobs) { k.RAMSize, k.DiskSize = "8Mi", "2Mi" }))

	short := shortScript(6 * time.Second)
	if short.GetLoopDurationMs() != 0 || len(short.GetOps()) != 1 {
		t.Fatalf("short script = %v, want one op and no loop", short)
	}
	if b := short.GetOps()[0].GetBurnCpu(); b.GetDurationMs() != 6000 || b.GetParallelism() != 1 {
		t.Errorf("short burn = %v, want 6000ms x1", b)
	}

	cpu := p.longScript(longCPU, time.Minute)
	if b := cpu.GetOps()[0].GetBurnCpu(); cpu.GetLoopDurationMs() != 0 || b.GetDurationMs() != 60000 || b.GetParallelism() != longCPUParallelism {
		t.Errorf("long cpu script = %v, want one 60000ms x%d burn", cpu, longCPUParallelism)
	}

	disk := p.longScript(longDisk, time.Minute)
	if disk.GetLoopDurationMs() != 60000 || len(disk.GetOps()) != 3 {
		t.Fatalf("long disk script = %v, want a 60000ms loop of 3 ops", disk)
	}
	if w := disk.GetOps()[0].GetWriteDisk(); w.GetKey() != diskKey || w.GetSize() != 2<<20 {
		t.Errorf("disk write = %v, want %s of 2Mi", w, diskKey)
	}
	if rd := disk.GetOps()[1].GetReadDisk(); rd.GetKey() != diskKey || rd.GetReadMode() != gluttonpb.ReadMode_READ_MODE_DIGEST_ONLY {
		t.Errorf("disk read = %v, want a digest-only read of %s", rd, diskKey)
	}
	if s := disk.GetOps()[2].GetSleep(); s.GetDurationMs() != loopPause.Milliseconds() {
		t.Errorf("disk pause = %v, want %v", s, loopPause)
	}

	ram := p.longScript(longRAM, 30*time.Second)
	if ram.GetLoopDurationMs() != 30000 || len(ram.GetOps()) != 3 {
		t.Fatalf("long ram script = %v, want a 30000ms loop of 3 ops", ram)
	}
	if w := ram.GetOps()[0].GetWriteRam(); w.GetKey() != ramKey || w.GetSize() != "8388608" || w.GetWriteMode() != gluttonpb.WriteMode_WRITE_MODE_OVERWRITE {
		t.Errorf("ram write = %v, want an OVERWRITE of 8Mi at %s", w, ramKey)
	}
	if rd := ram.GetOps()[1].GetReadRam(); rd.GetKey() != ramKey {
		t.Errorf("ram read = %v, want a walk of %s", rd, ramKey)
	}
}

// Ticks fall at epoch+phase and every interval after; the next one is
// always strictly ahead of now, so ticks missed during a session are gone.
func TestNextTick(t *testing.T) {
	epoch := time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)
	const phase, interval = 5 * time.Minute, 30 * time.Minute
	for _, tc := range []struct {
		now  time.Duration // after epoch
		want time.Duration
	}{
		{0, 5 * time.Minute},
		{5*time.Minute - time.Second, 5 * time.Minute},
		{5 * time.Minute, 35 * time.Minute},
		{20 * time.Minute, 35 * time.Minute},
		{2*time.Hour + 10*time.Minute, 2*time.Hour + 35*time.Minute},
	} {
		got := nextTick(epoch, phase, interval, epoch.Add(tc.now))
		if got != epoch.Add(tc.want) {
			t.Errorf("nextTick at +%v = +%v, want +%v", tc.now, got.Sub(epoch), tc.want)
		}
	}
}

// A dormant agent sleeps toward its tick in cronPollInterval slices, so a
// shorter interval stored mid-wait brings the tick forward.
func TestWaitForTickFollowsTheIntervalKnob(t *testing.T) {
	clk := &fakeClock{t: time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)}
	cfg := &userclass.Config{Dyn: dynconfig.Static(testKnobs(func(k *walkKnobs) { k.CronInterval = dynconfig.Seconds(time.Hour) }))}
	u := &agent{
		rt:        &runtime{now: clk.now, sleep: clk.sleep},
		cfg:       cfg,
		epoch:     clk.t,
		phaseFrac: 0.5, // tick at +30m under the hour, at +1m under the 2m set below
	}
	clk.onSleep = func(n int) {
		if n == 3 {
			cfg.Dyn = dynconfig.Static(testKnobs(func(k *walkKnobs) { k.CronInterval = dynconfig.Seconds(2 * time.Minute) }))
		}
	}
	u.waitForTick()
	// Three 10s slices under the hour-long interval, then the tick under
	// the 2m interval is at +1m: the wait ends there, not at +30m.
	if got := clk.t.Sub(u.epoch); got != time.Minute {
		t.Errorf("waited until +%v, want +1m", got)
	}
	// The next tick under 2m is at +3m: one 10s slice, then the remainder.
	u.waitForTick()
	if got := clk.t.Sub(u.epoch); got != 3*time.Minute {
		t.Errorf("second wait ended at +%v, want +3m", got)
	}
}

// One session end to end against the fakes, with weights that make the
// walk deterministic: short is the only weighted action, and the cap of
// one forces an LLM query (none has happened) and then done.
func TestSessionWalksShortThenLLMThenDone(t *testing.T) {
	srv := &fake.Server{}
	ctl := &fakeControlClient{}
	u := newTestAgent(t, srv, ctl, testKnobs(func(k *walkKnobs) {
		onlyWeights(0, 1, 0, 0)(k)
		k.MaxActions = 1
	}))

	u.step(context.Background()) // tick: wake + short compute
	if !u.active || u.actions != 1 || u.llmCalled {
		t.Fatalf("after step 1: active=%v actions=%d llmCalled=%v", u.active, u.actions, u.llmCalled)
	}
	scripts := srv.RecordedScripts()
	if len(scripts) != 1 || scripts[0].GetOps()[0].GetBurnCpu() == nil {
		t.Fatalf("scripts after step 1 = %v, want one burn", scripts)
	}
	if cpu := srv.RecordedCPURequests(); len(cpu) != 1 || cpu[0].GetNumCores() != 1 || cpu[0].GetDutyCycle() != 0.25 {
		t.Errorf("idle CPU requests = %v, want one of 1 core at 0.25", cpu)
	}

	u.step(context.Background()) // cap reached, LLM owed: suspend, think, wake
	if !u.llmCalled || u.actions != 2 || !u.active {
		t.Fatalf("after step 2: active=%v actions=%d llmCalled=%v", u.active, u.actions, u.llmCalled)
	}

	u.step(context.Background()) // cap reached, LLM done: done
	if u.active {
		t.Fatal("after step 3: still active")
	}

	wantPaths := []string{
		glutton.PingRoute,      // tick wake
		glutton.UseCPURoute,    // idle load, once
		glutton.RunScriptRoute, // short compute
		glutton.PingRoute,      // wake after the LLM think
	}
	if got := srv.RecordedPaths(); !slices.Equal(got, wantPaths) {
		t.Errorf("router paths = %v, want %v", got, wantPaths)
	}
	wantCalls := []string{"SuspendActor", "SuspendActor"} // before the LLM think, at done
	if got := ctl.recordedCalls(); !slices.Equal(got, wantCalls) {
		t.Errorf("control calls = %v, want %v", got, wantCalls)
	}
	if u.broken || u.consecutiveFailures != 0 {
		t.Errorf("clean session left broken=%v failures=%d", u.broken, u.consecutiveFailures)
	}

	// The next step waits out the cron and starts a new session.
	u.step(context.Background())
	if !u.active || u.actions != 1 {
		t.Errorf("after the next tick: active=%v actions=%d, want a fresh session", u.active, u.actions)
	}
}

// Explicit resume mode issues ResumeActor ahead of the wake ping, both at
// the tick and after an LLM think.
func TestExplicitResumeWakes(t *testing.T) {
	srv := &fake.Server{}
	ctl := &fakeControlClient{}
	u := newTestAgent(t, srv, ctl, testKnobs(func(k *walkKnobs) {
		k.ResumeMode = dynconfig.ResumeModeExplicit
		onlyWeights(1, 0, 0, 0)(k)
		k.MaxActions = 1
	}))
	u.step(context.Background()) // tick + LLM
	want := []string{"ResumeActor", "SuspendActor", "ResumeActor"}
	if got := ctl.recordedCalls(); !slices.Equal(got, want) {
		t.Errorf("control calls = %v, want %v", got, want)
	}
}

// A router that is busy fails the wake with a verdict that is not the
// actor's fault: the agent is kept, parked again, and left dormant until
// its next tick.
func TestWakeFailureKeepsActorThroughRouterCapacityErrors(t *testing.T) {
	ctl := &fakeControlClient{}
	u := newTestAgent(t, &fake.Server{Status: http.StatusServiceUnavailable}, ctl, testKnobs(nil))
	u.step(context.Background())
	if u.active || u.broken || u.consecutiveFailures != 0 {
		t.Errorf("after a 503 wake: active=%v broken=%v failures=%d, want dormant and kept", u.active, u.broken, u.consecutiveFailures)
	}
	if got := ctl.recordedCalls(); !slices.Equal(got, []string{"SuspendActor"}) {
		t.Errorf("control calls = %v, want the re-park only", got)
	}
}

// 404 from the router means the actor record is gone: replace at once.
func TestWakeFailureReplacesActorOnRouterNotFound(t *testing.T) {
	u := newTestAgent(t, &fake.Server{Status: http.StatusNotFound}, &fakeControlClient{}, testKnobs(nil))
	u.step(context.Background())
	if !u.broken {
		t.Error("broken = false after a 404 wake; want immediate replacement")
	}
}

// ateapi reports a CRASHED actor on ResumeActor; it never recovers, so the
// agent must replace it on the first failure.
func TestCrashedActorIsReplacedImmediately(t *testing.T) {
	ctl := &fakeControlClient{resumeErrs: []error{status.Error(codes.Aborted, "actor benchmark/walk-test crashed")}}
	u := newTestAgent(t, &fake.Server{}, ctl, testKnobs(func(k *walkKnobs) { k.ResumeMode = dynconfig.ResumeModeExplicit }))
	u.step(context.Background())
	if !u.broken || u.active {
		t.Errorf("after a crashed verdict: broken=%v active=%v, want true/false", u.broken, u.active)
	}
}

// A compute whose router reply is an error other than capacity counts
// toward replacement, and the session ends with the actor parked.
func TestComputeFailureEndsSessionAndCounts(t *testing.T) {
	srv := &fake.Server{}
	ctl := &fakeControlClient{}
	u := newTestAgent(t, srv, ctl, testKnobs(onlyWeights(0, 1, 0, 0)))
	u.step(context.Background()) // tick + a short compute that succeeds
	if !u.active {
		t.Fatal("first step did not start a session")
	}
	srv.Status = http.StatusInternalServerError
	u.step(context.Background())
	if u.active || u.consecutiveFailures != 1 || u.broken {
		t.Errorf("after a failed compute: active=%v failures=%d broken=%v, want dormant, 1, false", u.active, u.consecutiveFailures, u.broken)
	}
	if got := ctl.recordedCalls(); !slices.Equal(got, []string{"SuspendActor"}) {
		t.Errorf("control calls = %v, want one park at session end", got)
	}
}

// A retryable suspend failure strands the actor RUNNING. The next step
// must finish the suspend rather than wake an already-awake actor, which
// would book a few-ms WakeFirstTouch success.
func TestStrandedHibernateIsRedriven(t *testing.T) {
	srv := &fake.Server{}
	ctl := &fakeControlClient{suspendErrs: []error{status.Error(codes.Unavailable, "ate-api-server restarting")}}
	u := newTestAgent(t, srv, ctl, testKnobs(onlyWeights(1, 0, 0, 0)))

	u.step(context.Background()) // tick wake, then the LLM's suspend fails
	if u.active || !u.hibernatePending || u.broken {
		t.Fatalf("after failed suspend: active=%v hibernatePending=%v broken=%v, want false/true/false", u.active, u.hibernatePending, u.broken)
	}
	served := len(srv.RecordedPaths())

	u.step(context.Background()) // re-drive the suspend only
	if u.hibernatePending || u.active {
		t.Errorf("after re-drive: hibernatePending=%v active=%v, want false/false", u.hibernatePending, u.active)
	}
	if got := len(srv.RecordedPaths()); got != served {
		t.Errorf("router requests = %d, want %d (no wake against a stranded actor)", got, served)
	}
	if got := ctl.recordedCalls(); !slices.Equal(got, []string{"SuspendActor", "SuspendActor"}) {
		t.Errorf("control calls = %v, want two suspends", got)
	}
}

// Every control-plane RPC carries a deadline, so a server-side hang
// cannot park the VU goroutine for the rest of the run.
func TestControlRPCsCarryADeadline(t *testing.T) {
	ctl := &fakeControlClient{}
	u := newTestAgent(t, &fake.Server{}, ctl, testKnobs(nil))
	u.hibernate(context.Background())
	if !ctl.sawDeadline {
		t.Error("SuspendActor arrived without a deadline")
	}
}

// shutdown fans suspendAndDelete out across agents instead of sweeping
// them one at a time.
func TestShutdownFansOut(t *testing.T) {
	ctl := &fakeControlClient{deleteDelay: 20 * time.Millisecond}
	srv := &fake.Server{}
	ts := srv.Start(t)
	rt := newRuntime(&userclass.Config{
		APIStub: ctl, HTTPClient: ts.Client(), RouterURL: ts.URL, Atespace: "benchmark",
		Dyn: dynconfig.Static(testKnobs(nil)), Tracer: otel.Tracer("test"),
	})
	const n = 8
	for i := 0; i < n; i++ {
		rt.users.Store(i, &agent{rt: rt, cfg: rt.cfg, actorName: "walk-" + string(rune('a'+i))})
	}
	start := time.Now()
	rt.shutdown(context.Background())
	if elapsed := time.Since(start); elapsed > n*ctl.deleteDelay/2 {
		t.Errorf("shutdown took %v for %d agents; a serial sweep would take %v", elapsed, n, n*ctl.deleteDelay)
	}
	if got := ctl.maxInFlight.Load(); got < 2 {
		t.Errorf("max concurrent DeleteActor = %d, want > 1", got)
	}
}

// fakeClock stands in for time in waitForTick: sleep advances it.
type fakeClock struct {
	t       time.Time
	sleeps  int
	onSleep func(n int)
}

func (c *fakeClock) now() time.Time { return c.t }
func (c *fakeClock) sleep(d time.Duration) {
	c.t = c.t.Add(d)
	c.sleeps++
	if c.onSleep != nil {
		c.onSleep(c.sleeps)
	}
}

// newTestAgent builds an agent against srv and ctl whose clock does not
// wait: sleeps advance a fake clock, so think times and cron gaps are free.
func newTestAgent(t *testing.T, srv *fake.Server, ctl *fakeControlClient, dyn walkKnobs) *agent {
	t.Helper()
	ts := srv.Start(t)
	clk := &fakeClock{t: time.Date(2026, 10, 2, 12, 0, 0, 0, time.UTC)}
	rt := newRuntime(&userclass.Config{
		APIStub:    ctl,
		HTTPClient: ts.Client(),
		RouterURL:  ts.URL,
		Atespace:   "benchmark",
		Dyn:        dynconfig.Static(dyn),
		Tracer:     otel.Tracer("test"),
	})
	rt.now, rt.sleep = clk.now, clk.sleep
	return &agent{
		rt:        rt,
		cfg:       rt.cfg,
		actorName: "walk-test",
		template:  walkCodec.Defaults.Template,
		rng:       testRand(),
		epoch:     clk.t,
	}
}

type fakeControlClient struct {
	ateapipb.ControlClient
	mu          sync.Mutex
	calls       []string
	resumeErrs  []error
	suspendErrs []error
	// sawDeadline is set when a call arrived with a context deadline.
	sawDeadline bool
	// deleteDelay stalls each DeleteActor; inFlight and maxInFlight count
	// concurrent DeleteActor calls, to prove shutdown fans out.
	deleteDelay time.Duration
	inFlight    atomic.Int32
	maxInFlight atomic.Int32
}

func nextErr(errs *[]error) error {
	if len(*errs) == 0 {
		return nil
	}
	err := (*errs)[0]
	*errs = (*errs)[1:]
	return err
}

func (f *fakeControlClient) record(ctx context.Context, name string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, name)
	if _, ok := ctx.Deadline(); ok {
		f.sawDeadline = true
	}
}

func (f *fakeControlClient) CreateAtespace(ctx context.Context, in *ateapipb.CreateAtespaceRequest, opts ...grpc.CallOption) (*ateapipb.Atespace, error) {
	f.record(ctx, "CreateAtespace")
	return &ateapipb.Atespace{}, nil
}

func (f *fakeControlClient) CreateActor(ctx context.Context, in *ateapipb.CreateActorRequest, opts ...grpc.CallOption) (*ateapipb.Actor, error) {
	f.record(ctx, "CreateActor")
	return &ateapipb.Actor{}, nil
}

func (f *fakeControlClient) ResumeActor(ctx context.Context, in *ateapipb.ResumeActorRequest, opts ...grpc.CallOption) (*ateapipb.ResumeActorResponse, error) {
	f.record(ctx, "ResumeActor")
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := nextErr(&f.resumeErrs); err != nil {
		return nil, err
	}
	return &ateapipb.ResumeActorResponse{}, nil
}

func (f *fakeControlClient) SuspendActor(ctx context.Context, in *ateapipb.SuspendActorRequest, opts ...grpc.CallOption) (*ateapipb.SuspendActorResponse, error) {
	f.record(ctx, "SuspendActor")
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := nextErr(&f.suspendErrs); err != nil {
		return nil, err
	}
	return &ateapipb.SuspendActorResponse{}, nil
}

func (f *fakeControlClient) PauseActor(ctx context.Context, in *ateapipb.PauseActorRequest, opts ...grpc.CallOption) (*ateapipb.PauseActorResponse, error) {
	f.record(ctx, "PauseActor")
	return &ateapipb.PauseActorResponse{}, nil
}

func (f *fakeControlClient) DeleteActor(ctx context.Context, in *ateapipb.DeleteActorRequest, opts ...grpc.CallOption) (*ateapipb.Actor, error) {
	f.record(ctx, "DeleteActor")
	n := f.inFlight.Add(1)
	defer f.inFlight.Add(-1)
	for {
		cur := f.maxInFlight.Load()
		if n <= cur || f.maxInFlight.CompareAndSwap(cur, n) {
			break
		}
	}
	time.Sleep(f.deleteDelay)
	return &ateapipb.Actor{}, nil
}

func (f *fakeControlClient) recordedCalls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.calls...)
}
