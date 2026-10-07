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
	// ResidentRAM is the memory the agent holds from creation on, as a
	// Kubernetes quantity: the heap a real agent process keeps whether or
	// not it is doing anything. It is in every snapshot, and the long
	// operations churn within it.
	ResidentRAM string `json:"agentwalk_resident_ram"`
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
	// ShortMaxCores bounds the goroutines of a short burst; each burst
	// draws from 1 to this.
	ShortMaxCores int `json:"agentwalk_short_max_cores"`
	// The ceilings a long operation's mix is drawn under: goroutines for
	// its CPU part, bytes per cycle for its disk and RAM parts (Kubernetes
	// quantities). LongMaxRAM is also the resident working set every actor
	// ends up holding, see memoryFloor.
	LongMaxCores int    `json:"agentwalk_long_max_cores"`
	LongMaxDisk  string `json:"agentwalk_long_max_disk"`
	LongMaxRAM   string `json:"agentwalk_long_max_ram"`
	// LongCycleSeconds is the nominal length of one cycle of a long
	// operation; the operation repeats its cycle until its length is up.
	LongCycleSeconds dynconfig.Seconds `json:"agentwalk_long_cycle_seconds"`
	// LongJitter is how far each cycle's intensities stray from the
	// operation's draw, as a fraction in [0, 1).
	LongJitter float64 `json:"agentwalk_long_jitter"`
	// MaxActions is the count after which a session is wound up.
	MaxActions int `json:"agentwalk_max_actions"`
	// Template is the ActorTemplate in benchmark-workloads each agent is
	// created from.
	Template string `json:"agentwalk_template"`
}

var walkCodec = dynconfig.Typed[walkKnobs]{
	Defaults: walkKnobs{
		CronInterval:     dynconfig.Seconds(30 * time.Minute),
		IdleCPU:          0.25,
		ResidentRAM:      "64Mi",
		WeightLLM:        50,
		WeightShort:      30,
		WeightLong:       10,
		WeightDone:       10,
		ThinkMu:          2.0414,
		ThinkSigma:       0.8674,
		ShortSeconds:     dynconfig.Seconds(6 * time.Second),
		LongSeconds:      dynconfig.Seconds(time.Minute),
		ShortMaxCores:    1,
		LongMaxCores:     2,
		LongMaxDisk:      "16Mi",
		LongMaxRAM:       "32Mi",
		LongCycleSeconds: dynconfig.Seconds(2 * time.Second),
		LongJitter:       0.2,
		MaxActions:       50,
		Template:         "glutton",
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
	if _, err := parseSize(k.ResidentRAM); err != nil {
		return fmt.Errorf("agentwalk_resident_ram: %w", err)
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
	if k.ShortMaxCores < 1 {
		return fmt.Errorf("agentwalk_short_max_cores must be at least 1: %d", k.ShortMaxCores)
	}
	if k.LongMaxCores < 1 {
		return fmt.Errorf("agentwalk_long_max_cores must be at least 1: %d", k.LongMaxCores)
	}
	if _, err := parseSize(k.LongMaxRAM); err != nil {
		return fmt.Errorf("agentwalk_long_max_ram: %w", err)
	}
	diskBytes, err := parseSize(k.LongMaxDisk)
	if err != nil {
		return fmt.Errorf("agentwalk_long_max_disk: %w", err)
	}
	if diskBytes > math.MaxInt32 {
		return fmt.Errorf("agentwalk_long_max_disk cannot exceed %d (2 GiB), got: %d", math.MaxInt32, diskBytes)
	}
	if k.LongCycleSeconds <= 0 {
		return fmt.Errorf("agentwalk_long_cycle_seconds must be positive: %v", k.LongCycleSeconds.Duration())
	}
	if k.LongJitter < 0 || k.LongJitter >= 1 || math.IsNaN(k.LongJitter) {
		return fmt.Errorf("agentwalk_long_jitter must be in [0, 1), got: %f", k.LongJitter)
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

	// maxCycleVariants bounds how many distinct jittered cycles a long
	// operation's script spells out; the loop runs through them in turn, so
	// the jitter repeats with this period at most.
	maxCycleVariants = 16

	// minIOBytes is the least a disk or RAM part moves per cycle, so a
	// share drawn near zero still issues a real request rather than a
	// zero-byte one.
	minIOBytes = 4096

	// sandboxOverhead is the memory a glutton actor needs on top of the
	// working set a long operation makes resident: the guest, the glutton
	// process, and allocator transients. Observed around 115Mi for the
	// agentsession script; rounded up.
	sandboxOverhead = 128 << 20

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

// cyclePart is one of the four things a long operation's cycle spends
// its time on. The shares of a mix are indexed by it.
type cyclePart int

const (
	resCPU cyclePart = iota
	resDisk
	resRAM
	resIdle
	numParts
)

func (part cyclePart) String() string {
	switch part {
	case resCPU:
		return "cpu"
	case resDisk:
		return "disk"
	case resRAM:
		return "ram"
	case resIdle:
		return "idle"
	}
	return fmt.Sprintf("cyclePart(%d)", int(part))
}

// params is one activation's view of the knobs in the units the walk
// computes with: durations and byte counts rather than seconds and
// quantities. Re-read from dynconfig before every pick, so a knob change
// applies to the next action of every agent.
type params struct {
	cronInterval  time.Duration
	idleCPU       float64
	residentRAM   int64
	weights       [numActions]float64
	thinkMu       float64
	thinkSigma    float64
	shortMean     time.Duration
	longMean      time.Duration
	shortMaxCores int
	longMaxCores  int
	longMaxDisk   int64
	longMaxRAM    int64
	longCycle     time.Duration
	longJitter    float64
	maxActions    int
	template      string
}

// resolve converts knobs into params. The holder validated the knobs at
// store, so the sizes parse and the interval is positive; a holder built
// without validation (a test's) that breaks either falls back to that
// one default, since a zero interval would break the tick arithmetic.
func resolve(cfg walkKnobs) params {
	resolved := params{
		cronInterval:  cfg.CronInterval.Duration(),
		idleCPU:       cfg.IdleCPU,
		weights:       [numActions]float64{cfg.WeightLLM, cfg.WeightShort, cfg.WeightLong, cfg.WeightDone},
		thinkMu:       cfg.ThinkMu,
		thinkSigma:    cfg.ThinkSigma,
		shortMean:     cfg.ShortSeconds.Duration(),
		longMean:      cfg.LongSeconds.Duration(),
		shortMaxCores: max(cfg.ShortMaxCores, 1),
		longMaxCores:  max(cfg.LongMaxCores, 1),
		longCycle:     cfg.LongCycleSeconds.Duration(),
		longJitter:    cfg.LongJitter,
		maxActions:    cfg.MaxActions,
		template:      cfg.Template,
	}
	if resolved.cronInterval <= 0 {
		resolved.cronInterval = walkCodec.Defaults.CronInterval.Duration()
	}
	if resolved.longCycle <= 0 {
		resolved.longCycle = walkCodec.Defaults.LongCycleSeconds.Duration()
	}
	resolved.residentRAM, _ = parseSize(walkCodec.Defaults.ResidentRAM)
	if residentBytes, err := parseSize(cfg.ResidentRAM); err == nil {
		resolved.residentRAM = residentBytes
	}
	resolved.longMaxRAM, _ = parseSize(walkCodec.Defaults.LongMaxRAM)
	if ramBytes, err := parseSize(cfg.LongMaxRAM); err == nil {
		resolved.longMaxRAM = ramBytes
	}
	resolved.longMaxDisk, _ = parseSize(walkCodec.Defaults.LongMaxDisk)
	if diskBytes, err := parseSize(cfg.LongMaxDisk); err == nil && diskBytes <= math.MaxInt32 {
		resolved.longMaxDisk = diskBytes
	}
	return resolved
}

// memoryFloor is the least actor memory the walk is known to run under:
// the RAM array every actor holds (filled to residentRAM at creation and
// grown to longMaxRAM by a long operation if that is larger; it never
// shrinks), the scratch file (tmpfs), and the sandbox's own overhead.
func (p params) memoryFloor() int64 {
	return max(p.residentRAM, p.longMaxRAM) + p.longMaxDisk + sandboxOverhead
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

// shortCores draws the goroutines of a short burst, from 1 to the ceiling.
func (p params) shortCores(rng *rand.Rand) int32 {
	return int32(1 + rng.IntN(p.shortMaxCores))
}

// shortScript is a short compute burst: cores goroutines spinning for length.
func shortScript(length time.Duration, cores int32) *gluttonpb.RunScriptRequest {
	return script(&gluttonpb.Block{Steps: []*gluttonpb.Step{operation(burnRequest(length, cores))}})
}

// longMix is one long operation's draw: how a cycle is split between CPU,
// disk, RAM, and idling, and how hard each part goes. Every cycle of the
// operation follows it, so one operation is a consistent shape while the
// fleet's operations differ from one another.
type longMix struct {
	// shares sum to one. The CPU and idle shares are fractions of the
	// cycle's time; the disk and RAM shares are fractions of their byte
	// ceilings per cycle.
	shares [numParts]float64
	// cores is the CPU part's goroutine count.
	cores int32
}

// drawLongMix draws shares from a flat Dirichlet, by normalizing
// exponential draws, so the corners (all CPU, all idle) come up as readily
// as the middle, and cores uniformly up to the ceiling.
func (p params) drawLongMix(rng *rand.Rand) longMix {
	var mix longMix
	var total float64
	for i := range mix.shares {
		mix.shares[i] = -math.Log(1 - rng.Float64())
		total += mix.shares[i]
	}
	for i := range mix.shares {
		mix.shares[i] /= total
	}
	mix.cores = int32(1 + rng.IntN(p.longMaxCores))
	return mix
}

// longScript is one long operation lasting about length. Its first step
// brings the actor's RAM working set to the ceiling, so every actor holds
// the same resident set whatever its draws. Then a block loops over up to
// maxCycleVariants cycles until length is up. Each cycle is one operation
// running the CPU burn, a disk write and read, and a RAM churn and walk at
// the same time, sized by the mix and jittered independently per cycle,
// followed by the idle sleep. The cycles are spelled out rather than
// repeated so the jitter varies cycle to cycle.
func (p params) longScript(mix longMix, length time.Duration, rng *rand.Rand) *gluttonpb.RunScriptRequest {
	variants := int(length / p.longCycle)
	variants = max(1, min(variants, maxCycleVariants))
	cycles := make([]*gluttonpb.Step, 0, variants)
	for range variants {
		cycles = append(cycles, p.cycleStep(mix, rng))
	}
	return script(&gluttonpb.Block{Steps: []*gluttonpb.Step{
		operation(writeRAMRequest(p.longMaxRAM, gluttonpb.WriteMode_WRITE_MODE_OVERWRITE)),
		{Kind: &gluttonpb.Step_Block{Block: &gluttonpb.Block{
			LoopDurationMs: length.Milliseconds(),
			Steps:          cycles,
		}}},
	}})
}

// cycleStep is one jittered cycle of a long operation, as a block of three
// steps: the CPU burn alongside the disk and RAM writes, then the disk and
// RAM reads, then the idle sleep. The reads follow the writes rather than
// run beside them because an operation's requests run at the same time,
// and on an actor's first cycle the scratch file does not exist until its
// write lands. The block's result node reports the cycle's own elapsed time.
func (p params) cycleStep(mix longMix, rng *rand.Rand) *gluttonpb.Step {
	jitter := func() float64 { return 1 + p.longJitter*(2*rng.Float64()-1) }
	cycle := float64(p.longCycle)
	burn := time.Duration(mix.shares[resCPU] * cycle * jitter())
	idle := time.Duration(mix.shares[resIdle] * cycle * jitter())
	diskBytes := max(minIOBytes, int64(mix.shares[resDisk]*float64(p.longMaxDisk)*jitter()))
	ramBytes := max(minIOBytes, int64(mix.shares[resRAM]*float64(p.longMaxRAM)*jitter()))
	writes := operation(
		burnRequest(burn, mix.cores),
		&gluttonpb.Request{Kind: &gluttonpb.Request_WriteDisk{WriteDisk: &gluttonpb.WriteDiskRequest{
			Key: diskKey, Size: int32(min(diskBytes, math.MaxInt32)), WriteMode: gluttonpb.WriteMode_WRITE_MODE_TRUNCATE}}},
		// Rotate moves the dirty window through the working set cycle over
		// cycle instead of re-dirtying the same prefix.
		writeRAMRequest(ramBytes, gluttonpb.WriteMode_WRITE_MODE_OVERWRITE_ROTATE),
	)
	reads := operation(
		&gluttonpb.Request{Kind: &gluttonpb.Request_ReadDisk{ReadDisk: &gluttonpb.ReadDiskRequest{
			Key: diskKey, ReadMode: gluttonpb.ReadMode_READ_MODE_DIGEST_ONLY}}},
		&gluttonpb.Request{Kind: &gluttonpb.Request_ReadRam{ReadRam: &gluttonpb.ReadRAMRequest{
			Key: ramKey, Size: fmt.Sprintf("%d", ramBytes)}}},
	)
	return &gluttonpb.Step{Kind: &gluttonpb.Step_Block{Block: &gluttonpb.Block{
		Steps: []*gluttonpb.Step{writes, reads, operation(sleepRequest(idle))},
	}}}
}

func writeRAMRequest(bytes int64, mode gluttonpb.WriteMode) *gluttonpb.Request {
	return &gluttonpb.Request{Kind: &gluttonpb.Request_WriteRam{WriteRam: &gluttonpb.WriteRAMRequest{
		Key: ramKey, Size: fmt.Sprintf("%d", bytes), WriteMode: mode}}}
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
