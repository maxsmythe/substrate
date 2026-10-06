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
	"fmt"
	"math"
	"math/rand/v2"
	"net/http"
	"slices"
	"sort"
	"strings"
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
	knobs := resolve(walkCodec.Defaults)
	want := params{
		cronInterval:  30 * time.Minute,
		idleCPU:       0.25,
		weights:       [numActions]float64{50, 30, 10, 10},
		thinkMu:       2.0414,
		thinkSigma:    0.8674,
		shortMean:     6 * time.Second,
		longMean:      time.Minute,
		shortMaxCores: 1,
		longMaxCores:  2,
		longMaxDisk:   16 << 20,
		longMaxRAM:    32 << 20,
		longCycle:     2 * time.Second,
		longJitter:    0.2,
		maxActions:    50,
		template:      "glutton",
	}
	if knobs != want {
		t.Errorf("resolve(defaults) =\n %+v, want\n %+v", knobs, want)
	}
	if got := knobs.memoryFloor(); got != (32<<20)+(16<<20)+sandboxOverhead {
		t.Errorf("memoryFloor = %d, want ceilings plus overhead", got)
	}

	knobs = resolve(testKnobs(func(k *walkKnobs) {
		k.CronInterval = dynconfig.Seconds(5 * time.Minute)
		k.IdleCPU = 0.1
		k.LongMaxRAM = "1Gi"
		k.LongMaxDisk = "4Ki"
		k.LongMaxCores = 8
		k.LongCycleSeconds = dynconfig.Seconds(500 * time.Millisecond)
		k.Template = "glutton-big"
	}))
	if knobs.cronInterval != 5*time.Minute || knobs.idleCPU != 0.1 || knobs.longMaxRAM != 1<<30 || knobs.longMaxDisk != 4096 || knobs.longMaxCores != 8 || knobs.longCycle != 500*time.Millisecond || knobs.template != "glutton-big" {
		t.Errorf("set knobs did not carry: %+v", knobs)
	}
	// A holder built without validation can carry zeros; the arithmetic
	// that would divide by them falls back to the defaults.
	zeros := resolve(testKnobs(func(k *walkKnobs) { k.CronInterval = 0; k.LongCycleSeconds = 0 }))
	if zeros.cronInterval != 30*time.Minute || zeros.longCycle != 2*time.Second {
		t.Errorf("zero interval and cycle resolved to %v and %v, want the defaults", zeros.cronInterval, zeros.longCycle)
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
		"unparseable ram":    func(k *walkKnobs) { k.LongMaxRAM = "lots" },
		"zero disk":          func(k *walkKnobs) { k.LongMaxDisk = "0" },
		"zero short cores":   func(k *walkKnobs) { k.ShortMaxCores = 0 },
		"zero long cores":    func(k *walkKnobs) { k.LongMaxCores = 0 },
		"zero cycle":         func(k *walkKnobs) { k.LongCycleSeconds = 0 },
		"jitter of one":      func(k *walkKnobs) { k.LongJitter = 1 },
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
	rng := testRand()
	knobs := resolve(testKnobs(onlyWeights(1, 1, 1, 1000)))
	for i := 0; i < 1000; i++ {
		if picked := knobs.pick(rng, walkState{}); picked == actionDone {
			t.Fatal("done chosen as the first action")
		}
		if picked := knobs.pick(rng, walkState{actions: 5}); picked == actionDone {
			t.Fatal("done chosen before any LLM query")
		}
	}
	if picked := knobs.pick(rng, walkState{actions: 1, llmCalled: true}); picked != actionDone {
		t.Errorf("with done at weight 1000 and eligible, pick = %v", picked)
	}
	if picked := knobs.pick(rng, walkState{actions: knobs.maxActions}); picked != actionLLM {
		t.Errorf("at the cap without an LLM query, pick = %v, want llm", picked)
	}
	if picked := knobs.pick(rng, walkState{actions: knobs.maxActions + 1, llmCalled: true}); picked != actionDone {
		t.Errorf("at the cap after an LLM query, pick = %v, want done", picked)
	}
	// Only done has weight, and done is not allowed yet: the session ends
	// since nothing else may happen.
	onlyDone := resolve(testKnobs(onlyWeights(0, 0, 0, 1)))
	if picked := onlyDone.pick(rng, walkState{}); picked != actionDone {
		t.Errorf("with nothing eligible, pick = %v, want done", picked)
	}
}

// The draw follows the weights: a zero weight never comes up, and the
// shares of a long run land near the configured proportions.
func TestPickFollowsWeights(t *testing.T) {
	rng := testRand()
	knobs := resolve(testKnobs(onlyWeights(60, 40, 0, 0)))
	var counts [numActions]int
	const draws = 20000
	for i := 0; i < draws; i++ {
		counts[knobs.pick(rng, walkState{actions: 3, llmCalled: true})]++
	}
	if counts[actionLong] != 0 || counts[actionDone] != 0 {
		t.Errorf("zero-weight actions were drawn: %v", counts)
	}
	if share := float64(counts[actionLLM]) / draws; math.Abs(share-0.6) > 0.02 {
		t.Errorf("llm share = %.3f, want ~0.6", share)
	}
}

// The think time is log-normal with SWE-perf's moments: median near 7.7s,
// 90th percentile near 23.4s.
func TestThinkDistribution(t *testing.T) {
	rng := testRand()
	knobs := resolve(walkCodec.Defaults)
	const draws = 20000
	samples := make([]float64, draws)
	for i := range samples {
		samples[i] = knobs.think(rng).Seconds()
	}
	sort.Float64s(samples)
	if med := samples[draws/2]; math.Abs(med-7.7) > 0.4 {
		t.Errorf("median think = %.2fs, want ~7.7s", med)
	}
	if p90 := samples[draws*9/10]; math.Abs(p90-23.4) > 1.5 {
		t.Errorf("p90 think = %.2fs, want ~23.4s", p90)
	}
}

// Operation lengths stay within the spread around their means.
func TestDurationSpread(t *testing.T) {
	rng := testRand()
	knobs := resolve(testKnobs(func(k *walkKnobs) {
		k.ShortSeconds = dynconfig.Seconds(10 * time.Second)
		k.LongSeconds = dynconfig.Seconds(100 * time.Second)
	}))
	for i := 0; i < 1000; i++ {
		if length := knobs.shortDuration(rng); length < 5*time.Second || length > 15*time.Second {
			t.Fatalf("short duration %v outside [5s, 15s]", length)
		}
		if length := knobs.longDuration(rng); length < 50*time.Second || length > 150*time.Second {
			t.Fatalf("long duration %v outside [50s, 150s]", length)
		}
	}
}

// A short burst is one burn, with its goroutines drawn up to the ceiling.
func TestShortScript(t *testing.T) {
	short := shortScript(6*time.Second, 3)
	steps := short.GetScript().GetSteps()
	if short.GetScript().GetLoopDurationMs() != 0 || len(steps) != 1 {
		t.Fatalf("short script = %v, want one step and no loop", short)
	}
	burn := steps[0].GetOperation().GetRequests()[0].GetBurnCpu()
	if burn.GetDurationMs() != 6000 || burn.GetParallelism() != 3 {
		t.Errorf("short burn = %v, want 6000ms x3", burn)
	}
	rng := testRand()
	knobs := resolve(testKnobs(func(k *walkKnobs) { k.ShortMaxCores = 4 }))
	seen := map[int32]bool{}
	for i := 0; i < 200; i++ {
		cores := knobs.shortCores(rng)
		if cores < 1 || cores > 4 {
			t.Fatalf("shortCores = %d, outside [1, 4]", cores)
		}
		seen[cores] = true
	}
	if len(seen) != 4 {
		t.Errorf("shortCores drew %v, want every value in [1, 4]", seen)
	}
}

// A mix's shares sum to one and cover the simplex: over many draws every
// part is sometimes nearly all of the cycle and sometimes nearly none.
func TestDrawLongMix(t *testing.T) {
	rng := testRand()
	knobs := resolve(testKnobs(func(k *walkKnobs) { k.LongMaxCores = 3 }))
	var lo, hi [numParts]float64
	for i := range lo {
		lo[i] = 1
	}
	coresSeen := map[int32]bool{}
	for i := 0; i < 5000; i++ {
		mix := knobs.drawLongMix(rng)
		var total float64
		for part, share := range mix.shares {
			if share < 0 {
				t.Fatalf("negative share %v", mix.shares)
			}
			total += share
			lo[part] = min(lo[part], share)
			hi[part] = max(hi[part], share)
		}
		if math.Abs(total-1) > 1e-9 {
			t.Fatalf("shares %v sum to %v", mix.shares, total)
		}
		if mix.cores < 1 || mix.cores > 3 {
			t.Fatalf("cores = %d, outside [1, 3]", mix.cores)
		}
		coresSeen[mix.cores] = true
	}
	for part := cyclePart(0); part < numParts; part++ {
		if lo[part] > 0.02 || hi[part] < 0.8 {
			t.Errorf("%s share ranged [%.3f, %.3f] over 5000 draws; want the corners covered", part, lo[part], hi[part])
		}
	}
	if len(coresSeen) != 3 {
		t.Errorf("cores drew %v, want every value in [1, 3]", coresSeen)
	}
}

// A long operation's script brings the working set to the ceiling, then
// loops over jittered cycles until its length is up; each cycle is a block
// of the concurrent work followed by the idle sleep, sized by the mix.
func TestLongScript(t *testing.T) {
	knobs := resolve(testKnobs(func(k *walkKnobs) {
		k.LongMaxRAM, k.LongMaxDisk = "8Mi", "2Mi"
		k.LongCycleSeconds = dynconfig.Seconds(time.Second)
		k.LongJitter = 0.25
	}))
	mix := longMix{shares: [numParts]float64{0.4, 0.3, 0.2, 0.1}, cores: 2}
	req := knobs.longScript(mix, 30*time.Second, testRand())
	root := req.GetScript()
	if root.GetLoopDurationMs() != 0 || len(root.GetSteps()) != 2 {
		t.Fatalf("root = %v, want two unlooped steps", root)
	}
	if ws := root.GetSteps()[0].GetOperation().GetRequests()[0].GetWriteRam(); ws.GetKey() != ramKey || ws.GetSize() != "8388608" || ws.GetWriteMode() != gluttonpb.WriteMode_WRITE_MODE_OVERWRITE {
		t.Errorf("working-set step = %v, want an OVERWRITE of 8Mi at %s", ws, ramKey)
	}
	loop := root.GetSteps()[1].GetBlock()
	if loop.GetLoopDurationMs() != 30000 || len(loop.GetSteps()) != maxCycleVariants {
		t.Fatalf("loop = budget %d with %d cycles, want 30000 and %d", loop.GetLoopDurationMs(), len(loop.GetSteps()), maxCycleVariants)
	}
	within := func(got, nominal float64) bool { return got >= nominal*0.75-1 && got <= nominal*1.25+1 }
	distinct := map[int64]bool{}
	for i, step := range loop.GetSteps() {
		cycle := step.GetBlock()
		if cycle.GetLoopDurationMs() != 0 || len(cycle.GetSteps()) != 2 {
			t.Fatalf("cycle %d = %v, want an unlooped block of work then sleep", i, cycle)
		}
		work := cycle.GetSteps()[0].GetOperation().GetRequests()
		if len(work) != 5 {
			t.Fatalf("cycle %d work has %d requests, want burn, disk write, disk read, ram churn, ram walk", i, len(work))
		}
		burn := work[0].GetBurnCpu()
		if burn.GetParallelism() != 2 || !within(float64(burn.GetDurationMs()), 400) {
			t.Errorf("cycle %d burn = %v, want 2 goroutines for ~400ms", i, burn)
		}
		distinct[burn.GetDurationMs()] = true
		if write := work[1].GetWriteDisk(); write.GetKey() != diskKey || !within(float64(write.GetSize()), 0.3*(2<<20)) {
			t.Errorf("cycle %d disk write = %v, want ~30%% of 2Mi", i, write)
		}
		if read := work[2].GetReadDisk(); read.GetKey() != diskKey || read.GetReadMode() != gluttonpb.ReadMode_READ_MODE_DIGEST_ONLY {
			t.Errorf("cycle %d disk read = %v", i, read)
		}
		if churn := work[3].GetWriteRam(); churn.GetKey() != ramKey || churn.GetWriteMode() != gluttonpb.WriteMode_WRITE_MODE_OVERWRITE_ROTATE {
			t.Errorf("cycle %d ram churn = %v, want a rotate of %s", i, churn, ramKey)
		}
		if walk := work[4].GetReadRam(); walk.GetKey() != ramKey || walk.GetSize() != work[3].GetWriteRam().GetSize() {
			t.Errorf("cycle %d ram walk = %v, want the churned bytes walked", i, walk)
		}
		if sleep := cycle.GetSteps()[1].GetOperation().GetRequests()[0].GetSleep(); !within(float64(sleep.GetDurationMs()), 100) {
			t.Errorf("cycle %d sleep = %v, want ~100ms", i, sleep)
		}
	}
	if len(distinct) < 2 {
		t.Error("every cycle drew the same burn; the jitter is not applied per cycle")
	}
	// A short operation spells out fewer cycles, never zero.
	if got := len(knobs.longScript(mix, 1500*time.Millisecond, testRand()).GetScript().GetSteps()[1].GetBlock().GetSteps()); got != 1 {
		t.Errorf("a 1.5s operation at a 1s cycle spelled out %d cycles, want 1", got)
	}
	// Tiny shares still move something.
	tiny := knobs.longScript(longMix{shares: [numParts]float64{0.9998, 0.0001, 0.0001, 0}, cores: 1}, 5*time.Second, testRand())
	work := tiny.GetScript().GetSteps()[1].GetBlock().GetSteps()[0].GetBlock().GetSteps()[0].GetOperation().GetRequests()
	if work[1].GetWriteDisk().GetSize() < minIOBytes || work[3].GetWriteRam().GetSize() != fmt.Sprint(minIOBytes) {
		t.Errorf("tiny shares produced disk %d and ram %s, want the %d floor", work[1].GetWriteDisk().GetSize(), work[3].GetWriteRam().GetSize(), minIOBytes)
	}
}

// measure turns a result tree into per-leaf rates: bytes over the leaf's
// own elapsed, iterations per goroutine-second, slept over asked, and a
// cycle's elapsed per pass over its nominal length.
func TestMeasure(t *testing.T) {
	stats := func(elapsedMs, passes int64, edit func(s *gluttonpb.Stats)) *gluttonpb.StepResult {
		s := &gluttonpb.Stats{ElapsedMs: elapsedMs, Passes: passes}
		if edit != nil {
			edit(s)
		}
		return &gluttonpb.StepResult{Stats: s}
	}
	knobs := resolve(testKnobs(func(k *walkKnobs) { k.LongCycleSeconds = dynconfig.Seconds(time.Second) }))
	mix := longMix{shares: [numParts]float64{0.5, 0.2, 0.2, 0.1}, cores: 2}
	req := knobs.longScript(mix, 2*time.Second, testRand()) // two cycles spelled out
	leaf := func(kind string, elapsed, passes int64) *gluttonpb.StepResult {
		return stats(elapsed, passes, func(s *gluttonpb.Stats) {
			switch kind {
			// Sums over the five passes, as the glutton reports them.
			case "burn":
				s.BurnIterations = 4000 // 2 goroutines over 5s: 400 per core-second
			case "write":
				s.DiskBytesWritten = 20 << 20 // 20 MiB in 2.5s: 8 MiB/s
			case "read":
				s.DiskBytesRead = 10 << 20 // 10 MiB in 2.5s: 4 MiB/s
			case "churn":
				s.RamBytesWritten = 80 << 20 // 80 MiB in 2.5s: 32 MiB/s
			case "walk":
				s.RamBytesRead = 40 << 20 // 40 MiB in 2.5s: 16 MiB/s
			case "sleep":
				s.SleptMs = elapsed
			}
		})
	}
	// Both spelled-out cycles ran five passes each; the first cycle's work
	// took 1s and its sleep slept exactly what it asked; the second's
	// sleep overslept by half.
	cycleResult := func(cycleIndex int, oversleep float64) *gluttonpb.StepResult {
		askedMs := req.GetScript().GetSteps()[1].GetBlock().GetSteps()[cycleIndex].GetBlock().GetSteps()[1].GetOperation().GetRequests()[0].GetSleep().GetDurationMs()
		slept := int64(float64(askedMs) * 5 * oversleep)
		work := &gluttonpb.StepResult{Stats: &gluttonpb.Stats{ElapsedMs: 5000, Passes: 5}, Children: []*gluttonpb.StepResult{
			leaf("burn", 5000, 5), leaf("write", 2500, 5), leaf("read", 2500, 5), leaf("churn", 2500, 5), leaf("walk", 2500, 5),
		}}
		sleep := &gluttonpb.StepResult{Stats: &gluttonpb.Stats{ElapsedMs: slept, Passes: 5}, Children: []*gluttonpb.StepResult{leaf("sleep", slept, 5)}}
		return &gluttonpb.StepResult{Stats: &gluttonpb.Stats{ElapsedMs: 5000 + slept, Passes: 5}, Children: []*gluttonpb.StepResult{work, sleep}}
	}
	result := &gluttonpb.StepResult{Stats: &gluttonpb.Stats{Passes: 1}, Children: []*gluttonpb.StepResult{
		{Stats: &gluttonpb.Stats{Passes: 1}, Children: []*gluttonpb.StepResult{stats(0, 1, nil)}}, // working set, fake-like: no elapsed
		{Stats: &gluttonpb.Stats{Passes: 5}, Children: []*gluttonpb.StepResult{cycleResult(0, 1), cycleResult(1, 1.5)}},
	}}

	got := measure(req, result, time.Second)
	approx := func(name string, values []float64, want ...float64) {
		t.Helper()
		if len(values) != len(want) {
			t.Fatalf("%s = %v, want %v", name, values, want)
		}
		for i := range want {
			if math.Abs(values[i]-want[i]) > 0.02*want[i] {
				t.Errorf("%s = %v, want %v", name, values, want)
				return
			}
		}
	}
	approx("burn per core-second", got.burnPerCoreSecond, 400, 400)
	approx("disk write MiB/s", got.diskWriteMiBps, 8, 8)
	approx("disk read MiB/s", got.diskReadMiBps, 4, 4)
	// The working-set leaf had no elapsed and yields nothing; the churns do.
	approx("ram churn MiB/s", got.ramChurnMiBps, 32, 32)
	approx("ram walk MiB/s", got.ramWalkMiBps, 16, 16)
	approx("sleep ratio", got.sleepRatios, 1, 1.5)
	// Stretch is each cycle's elapsed per pass over the 1s nominal cycle.
	first := (5000 + float64(result.Children[1].Children[0].Children[1].Stats.ElapsedMs)) / 5 / 1000
	second := (5000 + float64(result.Children[1].Children[1].Children[1].Stats.ElapsedMs)) / 5 / 1000
	approx("cycle stretch", got.cycleStretches, first, second)
	// Without a cycle length there is no stretch, and rates still come out.
	if again := measure(req, result, 0); len(again.cycleStretches) != 0 || len(again.burnPerCoreSecond) != 2 {
		t.Errorf("measure without a cycle = %+v", again)
	}
	// A result shorter than the script (a cut pass) does not panic.
	measure(req, &gluttonpb.StepResult{Stats: &gluttonpb.Stats{}}, time.Second)
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
	walker := &agent{
		rt:        &runtime{now: clk.now, sleep: clk.sleep},
		cfg:       cfg,
		epoch:     clk.t,
		phaseFrac: 0.5, // tick at +30m under the hour, at +1m under the 2m set below
	}
	clk.onSleep = func(sleeps int) {
		if sleeps == 3 {
			cfg.Dyn = dynconfig.Static(testKnobs(func(k *walkKnobs) { k.CronInterval = dynconfig.Seconds(2 * time.Minute) }))
		}
	}
	walker.waitForTick()
	// Three 10s slices under the hour-long interval, then the tick under
	// the 2m interval is at +1m: the wait ends there, not at +30m.
	if got := clk.t.Sub(walker.epoch); got != time.Minute {
		t.Errorf("waited until +%v, want +1m", got)
	}
	// The next tick under 2m is at +3m: one 10s slice, then the remainder.
	walker.waitForTick()
	if got := clk.t.Sub(walker.epoch); got != 3*time.Minute {
		t.Errorf("second wait ended at +%v, want +3m", got)
	}
}

// One session end to end against the fakes, with weights that make the
// walk deterministic: short is the only weighted action, and the cap of
// one forces an LLM query (none has happened) and then done.
func TestSessionWalksShortThenLLMThenDone(t *testing.T) {
	srv := &fake.Server{}
	ctl := &fakeControlClient{}
	walker := newTestAgent(t, srv, ctl, testKnobs(func(k *walkKnobs) {
		onlyWeights(0, 1, 0, 0)(k)
		k.MaxActions = 1
	}))

	walker.step(context.Background()) // tick: wake + short compute
	if !walker.active || walker.actions != 1 || walker.llmCalled {
		t.Fatalf("after step 1: active=%v actions=%d llmCalled=%v", walker.active, walker.actions, walker.llmCalled)
	}
	scripts := srv.RecordedScripts()
	if len(scripts) != 1 || scripts[0].GetScript().GetSteps()[0].GetOperation().GetRequests()[0].GetBurnCpu() == nil {
		t.Fatalf("scripts after step 1 = %v, want one burn", scripts)
	}
	if cpu := srv.RecordedCPURequests(); len(cpu) != 1 || cpu[0].GetNumCores() != 1 || cpu[0].GetDutyCycle() != 0.25 {
		t.Errorf("idle CPU requests = %v, want one of 1 core at 0.25", cpu)
	}

	walker.step(context.Background()) // cap reached, LLM owed: suspend, think, wake
	if !walker.llmCalled || walker.actions != 2 || !walker.active {
		t.Fatalf("after step 2: active=%v actions=%d llmCalled=%v", walker.active, walker.actions, walker.llmCalled)
	}

	walker.step(context.Background()) // cap reached, LLM done: done
	if walker.active {
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
	if walker.broken || walker.consecutiveFailures != 0 {
		t.Errorf("clean session left broken=%v failures=%d", walker.broken, walker.consecutiveFailures)
	}

	// The next step waits out the cron and starts a new session.
	walker.step(context.Background())
	if !walker.active || walker.actions != 1 {
		t.Errorf("after the next tick: active=%v actions=%d, want a fresh session", walker.active, walker.actions)
	}
}

// Explicit resume mode issues ResumeActor ahead of the wake ping, both at
// the tick and after an LLM think.
func TestExplicitResumeWakes(t *testing.T) {
	srv := &fake.Server{}
	ctl := &fakeControlClient{}
	walker := newTestAgent(t, srv, ctl, testKnobs(func(k *walkKnobs) {
		k.ResumeMode = dynconfig.ResumeModeExplicit
		onlyWeights(1, 0, 0, 0)(k)
		k.MaxActions = 1
	}))
	walker.step(context.Background()) // tick + LLM
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
	walker := newTestAgent(t, &fake.Server{Status: http.StatusServiceUnavailable}, ctl, testKnobs(nil))
	walker.step(context.Background())
	if walker.active || walker.broken || walker.consecutiveFailures != 0 {
		t.Errorf("after a 503 wake: active=%v broken=%v failures=%d, want dormant and kept", walker.active, walker.broken, walker.consecutiveFailures)
	}
	if got := ctl.recordedCalls(); !slices.Equal(got, []string{"SuspendActor"}) {
		t.Errorf("control calls = %v, want the re-park only", got)
	}
}

// 404 from the router means the actor record is gone: replace at once.
func TestWakeFailureReplacesActorOnRouterNotFound(t *testing.T) {
	walker := newTestAgent(t, &fake.Server{Status: http.StatusNotFound}, &fakeControlClient{}, testKnobs(nil))
	walker.step(context.Background())
	if !walker.broken {
		t.Error("broken = false after a 404 wake; want immediate replacement")
	}
}

// ateapi reports a CRASHED actor on ResumeActor; it never recovers, so the
// agent must replace it on the first failure.
func TestCrashedActorIsReplacedImmediately(t *testing.T) {
	ctl := &fakeControlClient{resumeErrs: []error{status.Error(codes.Aborted, "actor benchmark/walk-test crashed")}}
	walker := newTestAgent(t, &fake.Server{}, ctl, testKnobs(func(k *walkKnobs) { k.ResumeMode = dynconfig.ResumeModeExplicit }))
	walker.step(context.Background())
	if !walker.broken || walker.active {
		t.Errorf("after a crashed verdict: broken=%v active=%v, want true/false", walker.broken, walker.active)
	}
}

// A compute whose router reply is an error other than capacity counts
// toward replacement, and the session ends with the actor parked.
func TestComputeFailureEndsSessionAndCounts(t *testing.T) {
	srv := &fake.Server{}
	ctl := &fakeControlClient{}
	walker := newTestAgent(t, srv, ctl, testKnobs(onlyWeights(0, 1, 0, 0)))
	walker.step(context.Background()) // tick + a short compute that succeeds
	if !walker.active {
		t.Fatal("first step did not start a session")
	}
	srv.Status = http.StatusInternalServerError
	walker.step(context.Background())
	if walker.active || walker.consecutiveFailures != 1 || walker.broken {
		t.Errorf("after a failed compute: active=%v failures=%d broken=%v, want dormant, 1, false", walker.active, walker.consecutiveFailures, walker.broken)
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
	walker := newTestAgent(t, srv, ctl, testKnobs(onlyWeights(1, 0, 0, 0)))

	walker.step(context.Background()) // tick wake, then the LLM's suspend fails
	if walker.active || !walker.hibernatePending || walker.broken {
		t.Fatalf("after failed suspend: active=%v hibernatePending=%v broken=%v, want false/true/false", walker.active, walker.hibernatePending, walker.broken)
	}
	served := len(srv.RecordedPaths())

	walker.step(context.Background()) // re-drive the suspend only
	if walker.hibernatePending || walker.active {
		t.Errorf("after re-drive: hibernatePending=%v active=%v, want false/false", walker.hibernatePending, walker.active)
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
	walker := newTestAgent(t, &fake.Server{}, ctl, testKnobs(nil))
	walker.hibernate(context.Background())
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
	const agents = 8
	for i := 0; i < agents; i++ {
		rt.users.Store(i, &agent{rt: rt, cfg: rt.cfg, actorName: "walk-" + string(rune('a'+i))})
	}
	start := time.Now()
	rt.shutdown(context.Background())
	if elapsed := time.Since(start); elapsed > agents*ctl.deleteDelay/2 {
		t.Errorf("shutdown took %v for %d agents; a serial sweep would take %v", elapsed, agents, agents*ctl.deleteDelay)
	}
	if got := ctl.maxInFlight.Load(); got < 2 {
		t.Errorf("max concurrent DeleteActor = %d, want > 1", got)
	}
}

// startUser refuses a template whose memory limit is below the working
// set the walk will make resident, and remembers the refusal briefly so a
// fleet of starting agents does not hammer ateapi.
func TestStartUserChecksTemplateMemory(t *testing.T) {
	srv := &fake.Server{}
	ctl := &fakeControlClient{templateMemory: "128Mi"}
	ts := srv.Start(t)
	rt := newRuntime(&userclass.Config{
		APIStub: ctl, HTTPClient: ts.Client(), RouterURL: ts.URL, Atespace: "benchmark",
		Dyn:    dynconfig.Static(testKnobs(nil)),
		Tracer: otel.Tracer("test"),
	})
	rt.now, rt.sleep = time.Now, func(time.Duration) {}

	if _, err := rt.startUser(context.Background()); err == nil || !strings.Contains(err.Error(), "below the walk's floor") {
		t.Fatalf("startUser against a 128Mi template: err = %v, want a floor refusal", err)
	}
	if _, err := rt.startUser(context.Background()); err == nil {
		t.Fatal("second startUser did not reuse the refusal")
	}
	if got := countCalls(ctl.recordedCalls(), "GetActorTemplate"); got != 1 {
		t.Errorf("GetActorTemplate called %d times, want 1 (the refusal is cached)", got)
	}
	if got := countCalls(ctl.recordedCalls(), "CreateActor"); got != 0 {
		t.Errorf("CreateActor called %d times against a refused template", got)
	}

	ctl.setTemplateMemory("1Gi")
	rt.templateErrAt = time.Time{} // let the cache expire
	walker, err := rt.startUser(context.Background())
	if err != nil {
		t.Fatalf("startUser against a 1Gi template: %v", err)
	}
	if walker == nil || countCalls(ctl.recordedCalls(), "CreateActor") != 1 {
		t.Errorf("startUser did not create the actor once the template fit: calls = %v", ctl.recordedCalls())
	}
	// Success is cached too.
	if _, err := rt.startUser(context.Background()); err != nil || countCalls(ctl.recordedCalls(), "GetActorTemplate") != 2 {
		t.Errorf("second successful startUser: err %v, GetActorTemplate calls %d, want 2", err, countCalls(ctl.recordedCalls(), "GetActorTemplate"))
	}
}

func countCalls(calls []string, name string) int {
	n := 0
	for _, c := range calls {
		if c == name {
			n++
		}
	}
	return n
}

// fakeClock stands in for time in waitForTick: sleep advances it.
type fakeClock struct {
	t       time.Time
	sleeps  int
	onSleep func(sleeps int)
}

func (c *fakeClock) now() time.Time { return c.t }
func (c *fakeClock) sleep(duration time.Duration) {
	c.t = c.t.Add(duration)
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
	// templateMemory is the memory limit GetActorTemplate reports; "" means
	// the template sets none.
	templateMemory string
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

func (f *fakeControlClient) setTemplateMemory(limit string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.templateMemory = limit
}

func (f *fakeControlClient) GetActorTemplate(ctx context.Context, in *ateapipb.GetActorTemplateRequest, opts ...grpc.CallOption) (*ateapipb.ActorTemplate, error) {
	f.record(ctx, "GetActorTemplate")
	f.mu.Lock()
	defer f.mu.Unlock()
	tmpl := &ateapipb.ActorTemplate{}
	if f.templateMemory != "" {
		tmpl.Resources = &ateapipb.Resources{Limits: []*ateapipb.Limits{{Name: "memory", Quantity: f.templateMemory}}}
	}
	return tmpl, nil
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
	concurrent := f.inFlight.Add(1)
	defer f.inFlight.Add(-1)
	for {
		cur := f.maxInFlight.Load()
		if concurrent <= cur || f.maxInFlight.CompareAndSwap(cur, concurrent) {
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
