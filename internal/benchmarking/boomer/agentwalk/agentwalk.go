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

// Package agentwalk implements the AgentWalkUser locust test: a fleet of
// personal agents of the Open Claw / Hermes kind, each one a glutton actor
// driven by a random walk rather than a fixed script, because no two such
// agents, and no two sessions of one agent, do the same work.
//
// Each VU is one agent with an internal cron. At every tick it wakes its
// actor and takes actions until it picks "done": an LLM query (the actor is
// suspended for a log-normal think time, then woken), a short compute burst
// (seconds of CPU, no suspend), or a long operation (about a minute of CPU,
// disk, or RAM work, no suspend). Done is never the first action and never
// comes before an LLM query. A tick that falls during a session is skipped.
// While awake the actor burns a small idle load, as the real agent does.
//
// The on-sandbox work goes through glutton's RunScript, one request per
// action, so the actor is held awake for exactly the activity's duration
// and the router sees one long request rather than many short ones.
package agentwalk

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/agent-substrate/substrate/internal/ateinterceptors"
	"github.com/agent-substrate/substrate/internal/atenet"
	"github.com/agent-substrate/substrate/internal/benchmarking/boomer/boomerutil"
	"github.com/agent-substrate/substrate/internal/benchmarking/boomer/dynconfig"
	bmetrics "github.com/agent-substrate/substrate/internal/benchmarking/boomer/metrics"
	"github.com/agent-substrate/substrate/internal/benchmarking/boomer/userclass"
	"github.com/agent-substrate/substrate/internal/benchmarking/glutton"
	gluttonpb "github.com/agent-substrate/substrate/internal/proto/glutton"
	"github.com/agent-substrate/substrate/pkg/proto/ateapipb"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

const (
	userClass = "AgentWalkUser"
	// templateNS is the atespace holding the benchmark ActorTemplates.
	templateNS = "benchmark-workloads"

	// controlRPCTimeout bounds every control-plane RPC, so a server-side
	// hang cannot park the VU goroutine for the rest of the run.
	controlRPCTimeout = 60 * time.Second

	// scriptSlack is the deadline headroom a RunScript gets beyond its
	// planned length: the wake it may trigger, the router, and the last
	// op of a loop running past the budget.
	scriptSlack = 30 * time.Second

	// cronPollInterval caps one sleep toward the next tick, so a change to
	// the interval knob reaches a dormant agent within this long.
	cronPollInterval = 10 * time.Second

	// maxConsecutiveFailures is how many ReplaceIfPersistent failures in a
	// row an agent tolerates before its actor is deleted and recreated.
	maxConsecutiveFailures = 3
)

// Stats rows. The lifecycle RPCs report under their own names.
const (
	wakeMetric      = "WakeFirstTouch"
	thinkMetric     = "LLMThink"
	shortMetric     = "ComputeShort"
	longMetricStem  = "ComputeLong_"
	sessionMetric   = "Session"
	idleCPUMetric   = "SetIdleCPU"
	crashMetric     = "CrashCount"
	methodHTTP      = "http"
	methodGRPC      = "grpc"
	methodActor     = "actor"
	wakePingMessage = "tick"
)

func init() {
	userclass.Add(userclass.Entry{
		Name:       "agentwalk",
		LocustFile: "agentwalk.py",
		UserClass:  userClass,
		Config:     walkCodec,
		Init:       initAgentWalk,
	})
}

func initAgentWalk(cfg *userclass.Config) (taskFn func(), shutdown func(context.Context)) {
	if cfg.Tracer == nil {
		cfg.Tracer = otel.Tracer("substrate-boomer/agentwalk")
	}
	rt := newRuntime(cfg)
	return rt.iterate, rt.shutdown
}

// runtime is the per-worker state shared by every boomer goroutine. Each
// goroutine keeps its own agent in users, keyed by goroutine ID, because
// boomer offers no per-VU context.
type runtime struct {
	cfg *userclass.Config
	// scriptClient sends RunScript requests. It shares the worker's
	// transport but has no client-wide timeout: a long operation outlives
	// the shared client's 30s, so each request is bounded by its own
	// deadline instead.
	scriptClient *http.Client
	users        sync.Map // goroutineID -> *agent

	// Clock hooks, replaced in tests so a session runs without waiting
	// out think times and cron gaps.
	now   func() time.Time
	sleep func(time.Duration)
}

func newRuntime(cfg *userclass.Config) *runtime {
	var transport http.RoundTripper
	if cfg.HTTPClient != nil {
		transport = cfg.HTTPClient.Transport
	}
	return &runtime{
		cfg:          cfg,
		scriptClient: &http.Client{Transport: transport},
		now:          time.Now,
		sleep:        time.Sleep,
	}
}

// iterate is the boomer task function: one action per call, preceded by a
// cron wait and a wake when the agent is dormant. The VU's agent is created
// lazily on first use and replaced when it is found broken.
func (r *runtime) iterate() {
	gid := boomerutil.GoroutineID()
	val, loaded := r.users.Load(gid)
	if !loaded {
		u, err := r.startUser(context.Background())
		if err != nil {
			slog.Warn("agentwalk start failed; goroutine will retry next iter",
				slog.String("err", err.Error()))
			r.sleep(2 * time.Second)
			return
		}
		val, _ = r.users.LoadOrStore(gid, u)
	}
	u := val.(*agent)

	u.step(context.Background())

	if u.broken {
		slog.Warn("agentwalk agent wedged; deleting its actor and starting a fresh one",
			slog.String("actor", u.actorName),
			slog.Int("consecutive_failures", u.consecutiveFailures))
		u.suspendAndDelete(context.Background())
		r.users.Delete(gid)
	}
}

// startUser creates the agent's actor, waits for its sandbox to serve,
// starts the idle load, and parks it so its first tick begins, like every
// later one, with a wake from suspension. The tick phase is drawn here, so
// a fleet started together spreads its ticks over the whole interval.
func (r *runtime) startUser(ctx context.Context) (*agent, error) {
	u := &agent{
		rt:        r,
		cfg:       r.cfg,
		actorName: "walk-" + uuid.NewString(),
		template:  resolve(dynconfig.Get[walkKnobs](r.cfg.Dyn)).template,
		rng:       rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64())),
		epoch:     r.now(),
	}
	u.phaseFrac = u.rng.Float64()
	slog.Info("Creating agentwalk agent", slog.String("actor", u.actorName), slog.String("template", u.template))
	bmetrics.UpdateUsers(userClass, 1)

	if err := u.ensureAtespace(ctx); err != nil {
		bmetrics.UpdateUsers(userClass, -1)
		return nil, fmt.Errorf("ensureAtespace: %w", err)
	}
	if err := u.create(ctx); err != nil {
		bmetrics.UpdateUsers(userClass, -1)
		return nil, fmt.Errorf("createActor: %w", err)
	}
	if err := u.waitServing(ctx); err != nil {
		// suspendAndDelete decrements the user gauge; no extra decrement here.
		u.suspendAndDelete(ctx)
		return nil, fmt.Errorf("waitServing: %w", err)
	}
	u.ensureIdleCPU(ctx)
	// A failed park is re-driven at the top of the first step.
	u.hibernate(ctx)
	return u, nil
}

// shutdownConcurrency bounds how many agents suspendAndDelete at once.
// Boomer gives the hook about a minute; at thousands of agents a serial
// sweep leaks most of the actors, while this many in flight clears them
// without flooding ateapi.
const shutdownConcurrency = 64

// shutdown suspends and deletes every agent's actor, shutdownConcurrency
// at a time, until done or ctx expires.
func (r *runtime) shutdown(ctx context.Context) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, shutdownConcurrency)
	r.users.Range(func(_, val any) bool {
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			return false
		}
		wg.Add(1)
		go func(u *agent) {
			defer wg.Done()
			defer func() { <-sem }()
			u.suspendAndDelete(ctx)
		}(val.(*agent))
		return true
	})
	wg.Wait()
}

// agent is one simulated personal agent: a single actor, its cron, and the
// session in progress. Owned by one boomer goroutine; no locking needed.
type agent struct {
	rt        *runtime
	cfg       *userclass.Config
	actorName string
	// template is the ActorTemplate the actor was created from, fixed at
	// creation; a knob change applies to agents created after it.
	template string
	rng      *rand.Rand

	// epoch and phaseFrac place the cron: ticks fall at
	// epoch + phaseFrac*interval and every interval after. The phase is a
	// fraction so a change to the interval keeps the fleet spread.
	epoch     time.Time
	phaseFrac float64

	// Session state. active is set from a successful tick wake until the
	// session ends; actions and llmCalled constrain the next pick.
	active       bool
	sessionStart time.Time
	actions      int
	llmCalled    bool

	idleCPUSet bool
	cleanedUp  bool
	// hibernatePending is set by a failed Pause/Suspend: the actor is
	// stranded RUNNING or SUSPENDING. Waking it from there would misreport
	// WakeFirstTouch, so step finishes the hibernate first.
	hibernatePending bool
	// consecutiveFailures counts ReplaceIfPersistent failures since the
	// last success; see noteFailure.
	consecutiveFailures int
	// broken is set once the actor should be replaced: a CRASHED or
	// otherwise stuck actor never recovers on its own, and without
	// replacement it would wedge its VU for the rest of the run.
	broken bool
}

// step runs one action. A dormant agent first waits for its cron tick and
// wakes its actor; the wake alone is one step when it fails.
func (u *agent) step(ctx context.Context) {
	// A failed hibernate left the actor awake. SuspendActor and PauseActor
	// are re-entrant, so finish it and pick the walk up next iteration.
	if u.hibernatePending {
		u.hibernate(ctx)
		return
	}
	if !u.active {
		u.waitForTick()
		if !u.wake(ctx) {
			return
		}
		u.active = true
		u.sessionStart = u.rt.now()
		u.actions = 0
		u.llmCalled = false
	}

	p := resolve(dynconfig.Get[walkKnobs](u.cfg.Dyn))
	a := p.pick(u.rng, walkState{actions: u.actions, llmCalled: u.llmCalled})
	if a != actionDone {
		u.actions++
	}
	switch a {
	case actionLLM:
		u.queryLLM(ctx, p)
	case actionShort:
		d := p.shortDuration(u.rng)
		u.compute(ctx, shortMetric, shortScript(d), d)
	case actionLong:
		kind := pickLongKind(u.rng)
		d := p.longDuration(u.rng)
		u.compute(ctx, longMetricStem+kind.String(), p.longScript(kind, d), d)
	case actionDone:
		u.endSession(ctx, nil)
	}
}

// waitForTick sleeps until the agent's next cron tick, re-reading the
// interval knob every cronPollInterval along the way.
func (u *agent) waitForTick() {
	for {
		interval := resolve(dynconfig.Get[walkKnobs](u.cfg.Dyn)).cronInterval
		now := u.rt.now()
		gap := nextTick(u.epoch, time.Duration(u.phaseFrac*float64(interval)), interval, now).Sub(now)
		if gap <= 0 {
			return
		}
		if gap > cronPollInterval {
			u.rt.sleep(cronPollInterval)
			continue
		}
		u.rt.sleep(gap)
		return
	}
}

// wake brings the actor back: ResumeActor first in explicit mode, then a
// ping that in implicit mode is itself the parked wake. The ping's latency
// is the WakeFirstTouch row. A failed wake hibernates the actor again so it
// is not left awake and idle until the next tick.
func (u *agent) wake(ctx context.Context) bool {
	if dynconfig.Get[walkKnobs](u.cfg.Dyn).ResumeMode == dynconfig.ResumeModeExplicit {
		if err := u.resume(ctx); err != nil {
			u.noteFailure(err)
			return false
		}
	}
	ctx, span := u.cfg.Tracer.Start(ctx, wakeMetric)
	defer span.End()
	start := time.Now()
	err := u.postProto(ctx, u.cfg.HTTPClient, glutton.PingRoute,
		&gluttonpb.PingRequest{Message: wakePingMessage}, &gluttonpb.PingResponse{})
	latency := time.Since(start)
	boomerutil.LogSampledTrace(span, wakeMetric, latency, boomerutil.SourceClient, err)
	if err != nil {
		u.noteFailure(err)
		bmetrics.RecordFailure(methodHTTP, wakeMetric, userClass, latency, err.Error())
		slog.Warn("agentwalk wake failed",
			slog.String("actor", u.actorName),
			slog.String("err", err.Error()))
		u.hibernate(ctx)
		return false
	}
	bmetrics.RecordSuccess(methodHTTP, wakeMetric, userClass, latency, 0)
	u.ensureIdleCPU(ctx)
	return true
}

// queryLLM suspends the actor for a think time and wakes it when the
// answer "arrives". The think time is its own row, so the distribution the
// fleet ran with can be read off the stats.
func (u *agent) queryLLM(ctx context.Context, p params) {
	if !u.hibernate(ctx) {
		u.endSession(ctx, errors.New("suspend before the LLM query failed"))
		return
	}
	think := p.think(u.rng)
	u.rt.sleep(think)
	bmetrics.RecordSuccess(methodActor, thinkMetric, userClass, think, 0)
	if !u.wake(ctx) {
		u.endSession(ctx, errors.New("wake after the LLM query failed"))
		return
	}
	u.llmCalled = true
}

// compute runs one RunScript against the actor and reports its wall time
// under name. The request is bounded by the script's planned length plus
// slack, on a client with no timeout of its own.
func (u *agent) compute(ctx context.Context, name string, script *gluttonpb.RunScriptRequest, planned time.Duration) {
	ctx, cancel := context.WithTimeout(ctx, planned+scriptSlack)
	defer cancel()
	ctx, span := u.cfg.Tracer.Start(ctx, name)
	defer span.End()
	start := time.Now()
	resp := &gluttonpb.RunScriptResponse{}
	err := u.postProto(ctx, u.rt.scriptClient, glutton.RunScriptRoute, script, resp)
	latency := time.Since(start)
	boomerutil.LogSampledTrace(span, name, latency, boomerutil.SourceClient, err)
	if err != nil {
		u.noteFailure(err)
		bmetrics.RecordFailure(methodHTTP, name, userClass, latency, err.Error())
		slog.Warn("agentwalk compute failed",
			slog.String("actor", u.actorName),
			slog.String("action", name),
			slog.String("err", err.Error()))
		u.endSession(ctx, err)
		return
	}
	u.noteSuccess()
	bmetrics.RecordSuccess(methodHTTP, name, userClass, latency, resp.GetOpsRun())
}

// endSession parks the actor until the next tick and books the session: a
// success row whose latency is the session's wall time and whose size is
// its action count, or a failure row carrying cause. The cause was already
// classified where it arose; a failed hibernate here leaves
// hibernatePending for the next step to re-drive.
func (u *agent) endSession(ctx context.Context, cause error) {
	if !u.hibernatePending {
		u.hibernate(ctx)
	}
	elapsed := u.rt.now().Sub(u.sessionStart)
	if cause != nil {
		bmetrics.RecordFailure(methodActor, sessionMetric, userClass, elapsed, cause.Error())
	} else {
		bmetrics.RecordSuccess(methodActor, sessionMetric, userClass, elapsed, int64(u.actions))
	}
	u.active = false
}

// ensureIdleCPU starts the agent's idle load: one goroutine at the
// configured duty cycle, through glutton UseCPU. The goroutine lives in
// the glutton process, so it survives suspend and resume and the actor
// draws the load whenever it is awake. Runs once per actor; a failure
// leaves idleCPUSet unset so the next wake retries.
func (u *agent) ensureIdleCPU(ctx context.Context) {
	if u.idleCPUSet {
		return
	}
	idle := resolve(dynconfig.Get[walkKnobs](u.cfg.Dyn)).idleCPU
	if idle <= 0 {
		u.idleCPUSet = true
		return
	}
	ctx, span := u.cfg.Tracer.Start(ctx, idleCPUMetric)
	defer span.End()
	start := time.Now()
	err := u.postProto(ctx, u.cfg.HTTPClient, glutton.UseCPURoute,
		&gluttonpb.UseCPURequest{NumCores: 1, DutyCycle: idle}, &gluttonpb.UseCPUResponse{})
	latency := time.Since(start)
	boomerutil.LogSampledTrace(span, idleCPUMetric, latency, boomerutil.SourceClient, err)
	if err != nil {
		bmetrics.RecordFailure(methodHTTP, idleCPUMetric, userClass, latency, err.Error())
		return
	}
	u.idleCPUSet = true
	bmetrics.RecordSuccess(methodHTTP, idleCPUMetric, userClass, latency, 0)
}

// httpError is a router reply with a status of 400 or above. It keeps the
// status so a failure can be classified the way a gRPC code would be: the
// router maps ateapi's ResourceExhausted and Unavailable to 503 and its
// deadline to 504, and in implicit resume mode the wake ping is the only
// place those verdicts reach the driver.
type httpError struct {
	route  string
	status int
	body   string
}

func (e *httpError) Error() string {
	return fmt.Sprintf("%s: HTTP %d: %s", e.route, e.status, e.body)
}

// classifyHTTP maps a router status onto a failure action. 503, 504, and
// 429 are the fleet or the control plane being busy, which a replacement
// would only add to; 404 means the actor record is gone. Anything else is
// counted toward replacement, as a non-status error would be.
func classifyHTTP(status int) boomerutil.FailureAction {
	switch status {
	case http.StatusServiceUnavailable, http.StatusGatewayTimeout, http.StatusTooManyRequests:
		return boomerutil.RetryLater
	case http.StatusNotFound:
		return boomerutil.ReplaceNow
	}
	return boomerutil.ReplaceIfPersistent
}

// noteFailure classifies a failed action or lifecycle call and marks the
// actor broken when a replacement is warranted. Cluster-wide errors (no
// capacity, ate-api-server restarting) are not the actor's fault and do
// not count, whether they arrive as a gRPC status or as the router's HTTP
// mapping of one.
func (u *agent) noteFailure(err error) {
	action := boomerutil.ClassifyLifecycleFailure(err)
	var he *httpError
	if errors.As(err, &he) {
		action = classifyHTTP(he.status)
	}
	switch action {
	case boomerutil.ReplaceNow:
		u.broken = true
	case boomerutil.RetryLater:
	default:
		u.consecutiveFailures++
		if u.consecutiveFailures >= maxConsecutiveFailures {
			u.broken = true
		}
	}
}

// noteSuccess clears the failure count: the actor just completed an
// action, so the earlier failures were transient after all.
func (u *agent) noteSuccess() {
	u.consecutiveFailures = 0
}

func (u *agent) ref() *ateapipb.ObjectRef {
	return &ateapipb.ObjectRef{Atespace: u.cfg.Atespace, Name: u.actorName}
}

// postProto POSTs req to the actor's route through the router with client
// and unmarshals the reply into resp. HTTP status >= 400 is an error
// carrying the response body.
func (u *agent) postProto(ctx context.Context, client *http.Client, route string, req, resp proto.Message) error {
	body, err := proto.Marshal(req)
	if err != nil {
		return err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, u.cfg.RouterURL+route, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set(atenet.TargetActorHeader, u.cfg.Atespace+"/"+u.actorName)
	httpReq.Header.Set("Content-Type", "application/x-protobuf")
	otel.GetTextMapPropagator().Inject(ctx, propagation.HeaderCarrier(httpReq.Header))

	httpResp, err := client.Do(httpReq)
	if err != nil {
		return err
	}
	defer httpResp.Body.Close()
	respBody, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return err
	}
	if httpResp.StatusCode >= 400 {
		return &httpError{route: route, status: httpResp.StatusCode, body: strings.TrimSpace(string(respBody))}
	}
	return proto.Unmarshal(respBody, resp)
}

// ensureAtespace creates the configured atespace, treating AlreadyExists as
// success so concurrent agents racing the first creation all proceed.
func (u *agent) ensureAtespace(ctx context.Context) error {
	return u.tracedCall(ctx, "CreateAtespace", func(callCtx context.Context, tr *metadata.MD) error {
		_, err := u.cfg.APIStub.CreateAtespace(callCtx, &ateapipb.CreateAtespaceRequest{
			Atespace: &ateapipb.Atespace{
				Metadata: &ateapipb.ResourceMetadata{Name: u.cfg.Atespace},
			},
		}, grpc.Trailer(tr))
		if s, ok := status.FromError(err); ok && s.Code() == codes.AlreadyExists {
			return nil
		}
		return err
	})
}

func (u *agent) create(ctx context.Context) error {
	return u.tracedCall(ctx, "CreateActor", func(callCtx context.Context, tr *metadata.MD) error {
		_, err := u.cfg.APIStub.CreateActor(callCtx, &ateapipb.CreateActorRequest{
			Actor: &ateapipb.Actor{
				Metadata:      &ateapipb.ResourceMetadata{Atespace: u.cfg.Atespace, Name: u.actorName},
				ActorTemplate: &ateapipb.ObjectRef{Atespace: templateNS, Name: u.template},
			},
		}, grpc.Trailer(tr))
		return err
	})
}

// waitServing pings the actor through the router until glutton answers. The
// first request is also what wakes a newly created actor, so this doubles
// as the initial activation.
func (u *agent) waitServing(ctx context.Context) error {
	const maxRetries = 30
	const retryInterval = 2 * time.Second
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		lastErr = u.postProto(ctx, u.cfg.HTTPClient, glutton.PingRoute,
			&gluttonpb.PingRequest{Message: "hello"}, &gluttonpb.PingResponse{})
		if lastErr == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(retryInterval):
		}
	}
	return fmt.Errorf("actor %s never served: %w", u.actorName, lastErr)
}

// resume issues ResumeActor, retrying concurrent-update conflicts inside
// the traced call so the reported latency spans every attempt.
func (u *agent) resume(ctx context.Context) error {
	err := u.tracedCall(ctx, "ResumeActor", func(callCtx context.Context, tr *metadata.MD) error {
		return boomerutil.RetryOnConflict(callCtx, func() error {
			_, err := u.cfg.APIStub.ResumeActor(callCtx, &ateapipb.ResumeActorRequest{Actor: u.ref()}, grpc.Trailer(tr))
			return err
		})
	})
	if err != nil {
		if boomerutil.IsCrashed(err) {
			bmetrics.RecordFailure(methodActor, crashMetric, userClass, 0, "actor entered ACTOR_STATE_CRASHED")
		}
		slog.Error("ResumeActor failed", slog.String("actor", u.actorName), slog.String("err", err.Error()))
	}
	return err
}

// hibernate suspends (or pauses, per LifecycleMode) the actor and reports
// whether it took. A failure leaves hibernatePending set so the next step
// re-drives it instead of waking a stranded actor.
func (u *agent) hibernate(ctx context.Context) bool {
	var err error
	if dynconfig.Get[walkKnobs](u.cfg.Dyn).LifecycleMode == dynconfig.LifecycleModePause {
		err = u.tracedCall(ctx, "PauseActor", func(callCtx context.Context, tr *metadata.MD) error {
			_, err := u.cfg.APIStub.PauseActor(callCtx, &ateapipb.PauseActorRequest{Actor: u.ref()}, grpc.Trailer(tr))
			return err
		})
	} else {
		err = u.tracedCall(ctx, "SuspendActor", func(callCtx context.Context, tr *metadata.MD) error {
			_, err := u.cfg.APIStub.SuspendActor(callCtx, &ateapipb.SuspendActorRequest{Actor: u.ref()}, grpc.Trailer(tr))
			return err
		})
	}
	u.hibernatePending = err != nil
	if err != nil {
		u.noteFailure(err)
	}
	return err == nil
}

// suspendAndDelete releases the actor and its worker. Suspend first: a
// worker only frees an actor it can account for, and an actor still awake
// at delete risks being left CRASHED. Safe to call more than once.
func (u *agent) suspendAndDelete(ctx context.Context) {
	if u.cleanedUp {
		return
	}
	suspendCtx, cancel := context.WithTimeout(ctx, controlRPCTimeout)
	_, _ = u.cfg.APIStub.SuspendActor(suspendCtx, &ateapipb.SuspendActorRequest{Actor: u.ref()})
	cancel()
	_ = u.tracedCall(ctx, "DeleteActor", func(callCtx context.Context, tr *metadata.MD) error {
		_, err := u.cfg.APIStub.DeleteActor(callCtx, &ateapipb.DeleteActorRequest{Actor: u.ref(), AnyState: true}, grpc.Trailer(tr))
		return err
	})
	bmetrics.UpdateUsers(userClass, -1)
	u.cleanedUp = true
}

// tracedCall runs one control-plane RPC under a span and a deadline, and
// records it as a locust stats row of the same name, preferring the
// server-measured elapsed time from the response trailer.
func (u *agent) tracedCall(ctx context.Context, name string, do func(context.Context, *metadata.MD) error) error {
	ctx, cancel := context.WithTimeout(ctx, controlRPCTimeout)
	defer cancel()
	ctx, span := u.cfg.Tracer.Start(ctx, name)
	defer span.End()

	start := time.Now()
	var tr metadata.MD
	err := do(ctx, &tr)
	latency := time.Since(start)

	serverLatency, source := boomerutil.ElapsedFromMD(tr, ateinterceptors.ServerElapsedTrailer, latency)
	boomerutil.LogSampledTrace(span, name, serverLatency, source, err)
	if err != nil {
		bmetrics.RecordFailure(methodGRPC, name, userClass, latency, err.Error())
		return err
	}
	bmetrics.RecordSuccess(methodGRPC, name, userClass, latency, 0)
	return nil
}
