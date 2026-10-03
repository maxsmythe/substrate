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
	"fmt"
	"math"
	"math/rand/v2"
	"time"

	"github.com/agent-substrate/substrate/internal/benchmarking/boomer/dynconfig"
	gluttonpb "github.com/agent-substrate/substrate/internal/proto/glutton"
	"k8s.io/apimachinery/pkg/api/resource"
)

// walkKnobs is the AgentWalkUser slice of the runtime config; the master
// populates the keys from the --agentwalk-* locust flags
// (common/agentwalk_config.py). The defaults below are what a key left
// unset reads as. The think-time moments come from SWE-perf's model of LLM
// latency: log-normal in seconds, which puts the median near 7.7s and the
// 90th percentile near 23.4s. The idle load is the figure measured for
// Open Claw.
type walkKnobs struct {
	dynconfig.Lifecycle
	// CronInterval is the gap between an agent's wakeup ticks.
	CronInterval dynconfig.Seconds `json:"agentwalk_cron_interval"`
	// IdleCPU is the fraction of one core the agent burns while awake and
	// idle, in [0, 1]; 0 turns the idle load off.
	IdleCPU float64 `json:"agentwalk_idle_cpu"`
	// The weights of the four actions when an activation picks one. Only
	// their proportions matter; a zero disables that action, and they
	// cannot all be zero.
	WeightLLM   float64 `json:"agentwalk_weight_llm"`
	WeightShort float64 `json:"agentwalk_weight_short"`
	WeightLong  float64 `json:"agentwalk_weight_long"`
	WeightDone  float64 `json:"agentwalk_weight_done"`
	// ThinkMu and ThinkSigma are the log-space moments of the log-normal
	// LLM think time, in seconds.
	ThinkMu    float64 `json:"agentwalk_think_mu"`
	ThinkSigma float64 `json:"agentwalk_think_sigma"`
	// ShortSeconds and LongSeconds are the mean lengths of a short compute
	// burst and a long operation; each draw is within durationSpread of
	// its mean.
	ShortSeconds dynconfig.Seconds `json:"agentwalk_short_seconds"`
	LongSeconds  dynconfig.Seconds `json:"agentwalk_long_seconds"`
	// RAMSize is the array a long RAM operation churns and DiskSize the
	// file a long disk operation rewrites, as Kubernetes quantities.
	RAMSize  string `json:"agentwalk_ram_size"`
	DiskSize string `json:"agentwalk_disk_size"`
	// MaxActions is the count after which a session is wound up.
	MaxActions int `json:"agentwalk_max_actions"`
	// Template is the ActorTemplate in benchmark-workloads each agent is
	// created from.
	Template string `json:"agentwalk_template"`
}

var walkCodec = dynconfig.Typed[walkKnobs]{
	Defaults: walkKnobs{
		CronInterval: dynconfig.Seconds(30 * time.Minute),
		IdleCPU:      0.25,
		WeightLLM:    50,
		WeightShort:  30,
		WeightLong:   10,
		WeightDone:   10,
		ThinkMu:      2.0414,
		ThinkSigma:   0.8674,
		ShortSeconds: dynconfig.Seconds(6 * time.Second),
		LongSeconds:  dynconfig.Seconds(time.Minute),
		RAMSize:      "32Mi",
		DiskSize:     "16Mi",
		MaxActions:   50,
		Template:     "glutton",
	},
	Validate: validateKnobs,
}

func validateKnobs(k walkKnobs) error {
	if err := k.Lifecycle.Validate(); err != nil {
		return err
	}
	if k.CronInterval <= 0 {
		return fmt.Errorf("agentwalk_cron_interval must be positive: %v", k.CronInterval.Duration())
	}
	if k.IdleCPU < 0 || k.IdleCPU > 1 {
		return fmt.Errorf("agentwalk_idle_cpu must be between 0.0 and 1.0, got: %f", k.IdleCPU)
	}
	var total float64
	for name, weight := range map[string]float64{
		"agentwalk_weight_llm":   k.WeightLLM,
		"agentwalk_weight_short": k.WeightShort,
		"agentwalk_weight_long":  k.WeightLong,
		"agentwalk_weight_done":  k.WeightDone,
	} {
		if weight < 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return fmt.Errorf("%s must be a non-negative number, got: %f", name, weight)
		}
		total += weight
	}
	if total <= 0 {
		return fmt.Errorf("the agentwalk_weight_* knobs cannot all be zero")
	}
	if math.IsNaN(k.ThinkMu) || math.IsInf(k.ThinkMu, 0) {
		return fmt.Errorf("agentwalk_think_mu must be a number, got: %f", k.ThinkMu)
	}
	if k.ThinkSigma <= 0 || math.IsNaN(k.ThinkSigma) || math.IsInf(k.ThinkSigma, 0) {
		return fmt.Errorf("agentwalk_think_sigma must be positive: %f", k.ThinkSigma)
	}
	if k.ShortSeconds <= 0 {
		return fmt.Errorf("agentwalk_short_seconds must be positive: %v", k.ShortSeconds.Duration())
	}
	if k.LongSeconds <= 0 {
		return fmt.Errorf("agentwalk_long_seconds must be positive: %v", k.LongSeconds.Duration())
	}
	if _, err := parseSize(k.RAMSize); err != nil {
		return fmt.Errorf("agentwalk_ram_size: %w", err)
	}
	diskBytes, err := parseSize(k.DiskSize)
	if err != nil {
		return fmt.Errorf("agentwalk_disk_size: %w", err)
	}
	if diskBytes > math.MaxInt32 {
		return fmt.Errorf("agentwalk_disk_size cannot exceed %d (2 GiB), got: %d", math.MaxInt32, diskBytes)
	}
	if k.MaxActions < 1 {
		return fmt.Errorf("agentwalk_max_actions must be at least 1: %d", k.MaxActions)
	}
	if k.Template == "" {
		return fmt.Errorf("agentwalk_template is required")
	}
	return nil
}

// parseSize reads a Kubernetes quantity ("32Mi") as a positive byte count.
func parseSize(text string) (int64, error) {
	quantity, err := resource.ParseQuantity(text)
	if err != nil {
		return 0, err
	}
	bytes, ok := quantity.AsInt64()
	if !ok || bytes <= 0 {
		return 0, fmt.Errorf("%q is not a positive whole byte count", text)
	}
	return bytes, nil
}

const (
	// durationSpread randomizes an operation's length: uniform over
	// mean*(1-spread) to mean*(1+spread).
	durationSpread = 0.5

	// longCPUParallelism is the goroutine count of a long CPU operation; a
	// build or test run uses more than one core, a short burst does not.
	longCPUParallelism = 2

	// loopPause paces the passes of a disk or RAM loop so the operation is
	// steady I/O for its whole length rather than one flat-out burst.
	loopPause = 100 * time.Millisecond

	// Sandbox object names the long operations reuse across sessions.
	ramKey  = "agentwalk_ws"
	diskKey = "agentwalk_scratch"
)

// action is what an activated agent chooses to do next.
type action int

const (
	// actionLLM: the agent asks the model and has nothing to do until the
	// answer arrives, so the actor is suspended for the think time.
	actionLLM action = iota
	// actionShort: a quick on-sandbox computation or a fast external API
	// call; a small CPU spike, no suspend.
	actionShort
	// actionLong: a build, a test run, a data crunch; about a minute of
	// CPU, disk, or RAM work, no suspend.
	actionLong
	// actionDone: the agent's tasks are finished and it goes dormant until
	// its next cron tick.
	actionDone
	numActions
)

func (a action) String() string {
	switch a {
	case actionLLM:
		return "llm"
	case actionShort:
		return "short"
	case actionLong:
		return "long"
	case actionDone:
		return "done"
	}
	return fmt.Sprintf("action(%d)", int(a))
}

// longKind is the resource a long operation leans on.
type longKind int

const (
	longCPU longKind = iota
	longDisk
	longRAM
	numLongKinds
)

func (k longKind) String() string {
	switch k {
	case longCPU:
		return "cpu"
	case longDisk:
		return "disk"
	case longRAM:
		return "ram"
	}
	return fmt.Sprintf("longKind(%d)", int(k))
}

// params is one activation's view of the knobs in the units the walk
// computes with: durations and byte counts rather than seconds and
// quantities. Re-read from dynconfig before every pick, so a knob change
// applies to the next action of every agent.
type params struct {
	cronInterval time.Duration
	idleCPU      float64
	weights      [numActions]float64
	thinkMu      float64
	thinkSigma   float64
	shortMean    time.Duration
	longMean     time.Duration
	ramSize      int64
	diskSize     int64
	maxActions   int
	template     string
}

// resolve converts knobs into params. The holder validated the knobs at
// store, so the sizes parse and the interval is positive; a holder built
// without validation (a test's) that breaks either falls back to that
// one default, since a zero interval would break the tick arithmetic.
func resolve(cfg walkKnobs) params {
	resolved := params{
		cronInterval: cfg.CronInterval.Duration(),
		idleCPU:      cfg.IdleCPU,
		weights:      [numActions]float64{cfg.WeightLLM, cfg.WeightShort, cfg.WeightLong, cfg.WeightDone},
		thinkMu:      cfg.ThinkMu,
		thinkSigma:   cfg.ThinkSigma,
		shortMean:    cfg.ShortSeconds.Duration(),
		longMean:     cfg.LongSeconds.Duration(),
		maxActions:   cfg.MaxActions,
		template:     cfg.Template,
	}
	if resolved.cronInterval <= 0 {
		resolved.cronInterval = walkCodec.Defaults.CronInterval.Duration()
	}
	resolved.ramSize, _ = parseSize(walkCodec.Defaults.RAMSize)
	if ramBytes, err := parseSize(cfg.RAMSize); err == nil {
		resolved.ramSize = ramBytes
	}
	resolved.diskSize, _ = parseSize(walkCodec.Defaults.DiskSize)
	if diskBytes, err := parseSize(cfg.DiskSize); err == nil && diskBytes <= math.MaxInt32 {
		resolved.diskSize = diskBytes
	}
	return resolved
}

// walkState is what constrains the next pick: how many actions the session
// has taken and whether the LLM has been consulted yet.
type walkState struct {
	actions   int
	llmCalled bool
}

// pick draws the next action. Done is never the first action and never
// comes before an LLM query; past maxActions the session is wound up, with
// one LLM query first if none has happened. Among the eligible actions the
// draw is proportional to the weights; if none of them has weight, the
// session ends, since there is nothing else it is allowed to do.
func (p params) pick(rng *rand.Rand, state walkState) action {
	if state.actions >= p.maxActions {
		if !state.llmCalled {
			return actionLLM
		}
		return actionDone
	}
	var total float64
	for a := action(0); a < numActions; a++ {
		if p.eligible(a, state) {
			total += p.weights[a]
		}
	}
	if total <= 0 {
		return actionDone
	}
	draw := rng.Float64() * total
	for a := action(0); a < numActions; a++ {
		if !p.eligible(a, state) {
			continue
		}
		draw -= p.weights[a]
		if draw < 0 {
			return a
		}
	}
	// Float rounding at the very top of the range lands on the last
	// eligible action, which the loop above would have returned had the
	// draw been a hair smaller.
	for a := numActions - 1; a >= 0; a-- {
		if p.eligible(a, state) && p.weights[a] > 0 {
			return a
		}
	}
	return actionDone
}

func (p params) eligible(candidate action, state walkState) bool {
	if candidate == actionDone {
		return state.actions > 0 && state.llmCalled
	}
	return true
}

// think draws an LLM round trip: log-normal in seconds.
func (p params) think(rng *rand.Rand) time.Duration {
	secs := math.Exp(p.thinkMu + p.thinkSigma*rng.NormFloat64())
	return time.Duration(secs * float64(time.Second))
}

// shortDuration and longDuration draw an operation's length around its mean.
func (p params) shortDuration(rng *rand.Rand) time.Duration { return spread(rng, p.shortMean) }
func (p params) longDuration(rng *rand.Rand) time.Duration  { return spread(rng, p.longMean) }

func spread(rng *rand.Rand, mean time.Duration) time.Duration {
	factor := 1 - durationSpread + 2*durationSpread*rng.Float64()
	return time.Duration(float64(mean) * factor)
}

// pickLongKind chooses the resource of a long operation, each as likely as
// the others.
func pickLongKind(rng *rand.Rand) longKind {
	return longKind(rng.IntN(int(numLongKinds)))
}

// shortScript is a short compute burst: one goroutine spinning for length.
func shortScript(length time.Duration) *gluttonpb.RunScriptRequest {
	return script(&gluttonpb.Block{Steps: []*gluttonpb.Step{operation(burnRequest(length, 1))}})
}

// longScript is a long operation of the given kind lasting about length.
// CPU is one burn across longCPUParallelism goroutines. Disk rewrites and
// re-reads a scratch file, RAM re-randomizes and walks a working set, each
// as a block that cycles write, read, pause until length elapses.
func (p params) longScript(kind longKind, length time.Duration) *gluttonpb.RunScriptRequest {
	switch kind {
	case longDisk:
		return script(&gluttonpb.Block{
			LoopDurationMs: length.Milliseconds(),
			Steps: []*gluttonpb.Step{
				operation(&gluttonpb.Request{Kind: &gluttonpb.Request_WriteDisk{WriteDisk: &gluttonpb.WriteDiskRequest{
					Key: diskKey, Size: int32(p.diskSize), WriteMode: gluttonpb.WriteMode_WRITE_MODE_TRUNCATE}}}),
				operation(&gluttonpb.Request{Kind: &gluttonpb.Request_ReadDisk{ReadDisk: &gluttonpb.ReadDiskRequest{
					Key: diskKey, ReadMode: gluttonpb.ReadMode_READ_MODE_DIGEST_ONLY}}}),
				operation(sleepRequest(loopPause)),
			},
		})
	case longRAM:
		// OVERWRITE grows the array on first touch and re-randomizes it in
		// place after, so the working set is allocated once per actor and
		// dirtied on every pass.
		return script(&gluttonpb.Block{
			LoopDurationMs: length.Milliseconds(),
			Steps: []*gluttonpb.Step{
				operation(&gluttonpb.Request{Kind: &gluttonpb.Request_WriteRam{WriteRam: &gluttonpb.WriteRAMRequest{
					Key: ramKey, Size: fmt.Sprintf("%d", p.ramSize), WriteMode: gluttonpb.WriteMode_WRITE_MODE_OVERWRITE}}}),
				operation(&gluttonpb.Request{Kind: &gluttonpb.Request_ReadRam{ReadRam: &gluttonpb.ReadRAMRequest{Key: ramKey}}}),
				operation(sleepRequest(loopPause)),
			},
		})
	default:
		return script(&gluttonpb.Block{Steps: []*gluttonpb.Step{operation(burnRequest(length, longCPUParallelism))}})
	}
}

func script(root *gluttonpb.Block) *gluttonpb.RunScriptRequest {
	return &gluttonpb.RunScriptRequest{Script: root}
}

// operation is a step whose requests run together.
func operation(requests ...*gluttonpb.Request) *gluttonpb.Step {
	return &gluttonpb.Step{Kind: &gluttonpb.Step_Operation{Operation: &gluttonpb.Operation{Requests: requests}}}
}

func burnRequest(length time.Duration, parallelism int32) *gluttonpb.Request {
	return &gluttonpb.Request{Kind: &gluttonpb.Request_BurnCpu{BurnCpu: &gluttonpb.BurnCPURequest{
		DurationMs: length.Milliseconds(), Parallelism: parallelism}}}
}

func sleepRequest(length time.Duration) *gluttonpb.Request {
	return &gluttonpb.Request{Kind: &gluttonpb.Request_Sleep{Sleep: &gluttonpb.SleepRequest{DurationMs: length.Milliseconds()}}}
}

// nextTick is the first cron tick strictly after now. Ticks fall at
// epoch+phase and every interval after; computing the next one from now
// rather than from the last tick taken is what skips the ticks that fell
// while the agent was busy.
func nextTick(epoch time.Time, phase, interval time.Duration, now time.Time) time.Time {
	first := epoch.Add(phase)
	if now.Before(first) {
		return first
	}
	elapsed := now.Sub(first)
	ticksPast := elapsed/interval + 1
	return first.Add(ticksPast * interval)
}
