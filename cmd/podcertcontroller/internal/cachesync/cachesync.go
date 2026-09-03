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

// Package cachesync waits for informer caches the way cache.WaitForCacheSync
// does, but is never silent about a stall.
package cachesync

import (
	"context"
	"log/slog"
	"time"

	"k8s.io/client-go/tools/cache"
)

// warnAfter is how long a sync may take before Wait starts reporting it, and
// the interval between reports after that.
var warnAfter = 15 * time.Second

// Wait blocks until every synced func reports true or ctx is done, and
// returns whether the caches synced. Once the wait exceeds warnAfter it logs
// the elapsed time at that interval, and logs once more when a slow sync
// completes.
//
// client-go logs a LIST or WATCH that fails, but a request the API server
// accepts and never answers (one held in an API Priority and Fairness queue,
// for instance) blocks inside the transport with no log at all, and a
// controller waiting on it looks healthy while doing nothing.
func Wait(ctx context.Context, informer string, synced ...cache.InformerSynced) bool {
	return wait(ctx, informer, warnAfter, synced...)
}

func wait(ctx context.Context, informer string, warnAfter time.Duration, synced ...cache.InformerSynced) bool {
	start := time.Now()
	done := make(chan bool, 1)
	go func() { done <- cache.WaitForCacheSync(ctx.Done(), synced...) }()

	ticker := time.NewTicker(warnAfter)
	defer ticker.Stop()
	warned := false
	for {
		select {
		case ok := <-done:
			if warned {
				slog.InfoContext(ctx, "informer cache sync finished",
					slog.String("informer", informer),
					slog.Bool("synced", ok),
					slog.Duration("elapsed", time.Since(start)))
			}
			return ok
		case <-ticker.C:
			warned = true
			slog.WarnContext(ctx, "informer cache not yet synced",
				slog.String("informer", informer),
				slog.Duration("elapsed", time.Since(start)))
		}
	}
}
