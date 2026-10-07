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
	"time"

	bmetrics "github.com/agent-substrate/substrate/internal/benchmarking/boomer/metrics"
	gluttonpb "github.com/agent-substrate/substrate/internal/proto/glutton"
	"github.com/prometheus/client_golang/prometheus"
)

// The contention signals. A script's result tree carries each request's
// own elapsed time next to what it did, so every leaf yields a rate that
// an idle sandbox achieves at a known value and a contended one falls
// short of, whatever the mix was. Pass counts under a budget say far
// less: a burn is fixed wall-clock, the idle sleep pads every cycle, and
// the I/O parts only register once they outlast the burn beside them.
var (
	burnRate = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "agentwalk_burn_iterations_per_core_second",
		Help:    "BurnCPU sha256 iterations per goroutine-second; drops when the sandbox is short of CPU while the burn's wall clock stays fixed.",
		Buckets: prometheus.ExponentialBuckets(256, 2, 16),
	})
	diskWriteRate = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "agentwalk_disk_write_mib_per_second",
		Help:    "WriteDisk MiB/s per request inside long operations.",
		Buckets: prometheus.ExponentialBuckets(1, 2, 14),
	})
	diskReadRate = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "agentwalk_disk_read_mib_per_second",
		Help:    "ReadDisk MiB/s per request inside long operations.",
		Buckets: prometheus.ExponentialBuckets(1, 2, 14),
	})
	ramChurnRate = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "agentwalk_ram_churn_mib_per_second",
		Help:    "WriteRAM MiB/s per request inside long operations.",
		Buckets: prometheus.ExponentialBuckets(1, 2, 16),
	})
	ramWalkRate = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "agentwalk_ram_walk_mib_per_second",
		Help:    "ReadRAM MiB/s per request inside long operations; after a resume this is the demand-paging rate.",
		Buckets: prometheus.ExponentialBuckets(1, 2, 16),
	})
	sleepRatio = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "agentwalk_sleep_ratio",
		Help:    "Time slept over time asked for; above 1 the sandbox is not getting scheduled when its timer fires.",
		Buckets: []float64{0.9, 1, 1.05, 1.1, 1.25, 1.5, 2, 4, 8},
	})
	cycleStretch = prometheus.NewHistogram(prometheus.HistogramOpts{
		Name:    "agentwalk_cycle_stretch",
		Help:    "A long operation's cycle elapsed over its nominal length. At or under 1 the I/O parts fit the time the mix left them; above 1 the sandbox is behind.",
		Buckets: []float64{0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 8},
	})
	longShare = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "agentwalk_long_share",
		Help:    "The share of a long operation's cycle drawn for each resource, so the distribution the fleet ran with can be read off.",
		Buckets: prometheus.LinearBuckets(0.05, 0.1, 10),
	}, []string{"resource"})
)

func init() {
	prometheus.MustRegister(burnRate, diskWriteRate, diskReadRate, ramChurnRate, ramWalkRate, sleepRatio, cycleStretch, longShare)
}

// rates is what one script's result yields: the rate samples for the
// histograms and a locust row per leaf, kept apart from the recording so
// a test can check the arithmetic.
type rates struct {
	burnPerCoreSecond []float64
	diskWriteMiBps    []float64
	diskReadMiBps     []float64
	ramChurnMiBps     []float64
	ramWalkMiBps      []float64
	sleepRatios       []float64
	cycleStretches    []float64
	rows              []leafRow
}

// leafRow is one leaf of a script result as a locust row: the request's
// kind and the action's length make the name (DiskWriteLong, BurnShort),
// and a looped leaf's sums are spread back over its passes, so latency and
// length are per request and count is how many of them ran.
type leafRow struct {
	name    string
	latency time.Duration
	length  int64
	count   int64
}

// measure walks a script and its result side by side and computes the
// rate of every leaf. A node inside a looped block carries sums over its
// passes, so bytes over elapsed is that leaf's average rate across the
// operation. cycle is the nominal cycle length for the stretch of each
// block directly under a looped block; zero skips the stretch. action is
// the suffix of the leaf rows' names.
func measure(req *gluttonpb.RunScriptRequest, result *gluttonpb.StepResult, cycle time.Duration, action string) rates {
	out := rates{}
	measureBlock(req.GetScript(), result, cycle, false, action, &out)
	return out
}

func measureBlock(block *gluttonpb.Block, result *gluttonpb.StepResult, cycle time.Duration, underLoop bool, action string, out *rates) {
	if block == nil || result == nil {
		return
	}
	if underLoop && cycle > 0 && result.GetStats().GetPasses() > 0 {
		perPass := float64(result.GetStats().GetElapsedMs()) / float64(result.GetStats().GetPasses())
		out.cycleStretches = append(out.cycleStretches, perPass/float64(cycle.Milliseconds()))
	}
	children := result.GetChildren()
	for i, step := range block.GetSteps() {
		if i >= len(children) {
			return
		}
		switch kind := step.GetKind().(type) {
		case *gluttonpb.Step_Block:
			measureBlock(kind.Block, children[i], cycle, block.GetLoopDurationMs() > 0, action, out)
		case *gluttonpb.Step_Operation:
			requests := kind.Operation.GetRequests()
			leaves := children[i].GetChildren()
			for j, request := range requests {
				if j >= len(leaves) {
					break
				}
				measureRequest(request, leaves[j].GetStats(), action, out)
			}
		}
	}
}

// measureRequest turns one leaf's stats into its rate sample and its row.
// A leaf the fake actor answered, or one cut before it did anything, has
// no elapsed time and yields nothing.
func measureRequest(request *gluttonpb.Request, stats *gluttonpb.Stats, action string, out *rates) {
	elapsed := float64(stats.GetElapsedMs()) / 1000
	passes := stats.GetPasses()
	if elapsed <= 0 || passes <= 0 {
		return
	}
	mibps := func(bytes int64) float64 { return float64(bytes) / (1 << 20) / elapsed }
	var name string
	var length int64
	switch kind := request.GetKind().(type) {
	case *gluttonpb.Request_BurnCpu:
		goroutines := max(float64(kind.BurnCpu.GetParallelism()), 1)
		out.burnPerCoreSecond = append(out.burnPerCoreSecond, float64(stats.GetBurnIterations())/goroutines/elapsed)
		name, length = "Burn", stats.GetBurnIterations()
	case *gluttonpb.Request_WriteDisk:
		out.diskWriteMiBps = append(out.diskWriteMiBps, mibps(stats.GetDiskBytesWritten()))
		name, length = "DiskWrite", stats.GetDiskBytesWritten()
	case *gluttonpb.Request_Ingest:
		out.diskWriteMiBps = append(out.diskWriteMiBps, mibps(stats.GetDiskBytesWritten()))
		name, length = "Ingest", stats.GetDiskBytesWritten()
	case *gluttonpb.Request_ReadDisk:
		out.diskReadMiBps = append(out.diskReadMiBps, mibps(stats.GetDiskBytesRead()))
		name, length = "DiskRead", stats.GetDiskBytesRead()
	case *gluttonpb.Request_WriteRam:
		out.ramChurnMiBps = append(out.ramChurnMiBps, mibps(stats.GetRamBytesWritten()))
		// The one-time fill to the ceiling and the per-cycle rotate are
		// different work; the mode tells them apart.
		name, length = "RAMChurn", stats.GetRamBytesWritten()
		if kind.WriteRam.GetWriteMode() != gluttonpb.WriteMode_WRITE_MODE_OVERWRITE_ROTATE {
			name = "RAMWrite"
		}
	case *gluttonpb.Request_ReadRam:
		out.ramWalkMiBps = append(out.ramWalkMiBps, mibps(stats.GetRamBytesRead()))
		name, length = "RAMWalk", stats.GetRamBytesRead()
	case *gluttonpb.Request_Sleep:
		// Summed over passes like everything else, so the ratio is the
		// average; a sleep the budget never let run has no asked time.
		asked := float64(kind.Sleep.GetDurationMs()) * float64(passes)
		if asked > 0 {
			out.sleepRatios = append(out.sleepRatios, float64(stats.GetSleptMs())/asked)
		}
		name = "Sleep"
	default:
		return
	}
	out.rows = append(out.rows, leafRow{
		name:    name + action,
		latency: time.Duration(stats.GetElapsedMs()/passes) * time.Millisecond,
		length:  length / passes,
		count:   passes,
	})
}

// methodScript marks the leaf rows: work the glutton did inside a script,
// as opposed to a request the driver made.
const methodScript = "script"

// observe records a script's rates and its leaf rows, one locust
// observation per request run.
func (r rates) observe() {
	for _, row := range r.rows {
		for range row.count {
			bmetrics.RecordSuccess(methodScript, row.name, userClass, row.latency, row.length)
		}
	}
	for _, v := range r.burnPerCoreSecond {
		burnRate.Observe(v)
	}
	for _, v := range r.diskWriteMiBps {
		diskWriteRate.Observe(v)
	}
	for _, v := range r.diskReadMiBps {
		diskReadRate.Observe(v)
	}
	for _, v := range r.ramChurnMiBps {
		ramChurnRate.Observe(v)
	}
	for _, v := range r.ramWalkMiBps {
		ramWalkRate.Observe(v)
	}
	for _, v := range r.sleepRatios {
		sleepRatio.Observe(v)
	}
	for _, v := range r.cycleStretches {
		cycleStretch.Observe(v)
	}
}

// observeMix records the shares a long operation was drawn with.
func observeMix(mix longMix) {
	for part := cyclePart(0); part < numParts; part++ {
		longShare.WithLabelValues(part.String()).Observe(mix.shares[part])
	}
}
