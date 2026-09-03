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

package rendezvous

import (
	"context"
	"fmt"
	"testing"
	"time"

	coordinationv1 "k8s.io/api/coordination/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/clock"
	"k8s.io/utils/ptr"
)

const (
	testNS   = "podcertificate-controller-system"
	testApp  = "podcertificate-controller"
	testSelf = "podcert-self"
	testItem = "maintain-trust-bundles"
)

// newTestHasher returns a Hasher whose informer is never run; tests seed the
// indexer directly, which is all AssignedToThisReplica reads.
func newTestHasher(t *testing.T) *Hasher {
	t.Helper()
	return New(fake.NewSimpleClientset(), testNS, testApp, testSelf, "uid-self", clock.RealClock{})
}

// addLease seeds a lease for holder renewed at renewedAt with the default
// lease duration. A nil holder or renewedAt leaves that field unset.
func addLease(t *testing.T, h *Hasher, name string, holder *string, renewedAt *time.Time) {
	t.Helper()
	lease := &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testNS,
			Name:      name,
			Labels:    map[string]string{labelKey: testApp},
		},
		Spec: coordinationv1.LeaseSpec{
			HolderIdentity:       holder,
			LeaseDurationSeconds: ptr.To(int32(leaseDuration / time.Second)),
		},
	}
	if renewedAt != nil {
		lease.Spec.RenewTime = ptr.To(metav1.NewMicroTime(*renewedAt))
	}
	if err := h.leaseInformer.GetIndexer().Add(lease); err != nil {
		t.Fatalf("seeding lease %q: %v", name, err)
	}
}

// winningPeer returns a replica name that beats testSelf in the rendezvous
// hash for testItem, so a test can prove a peer was excluded rather than
// merely outvoted.
func winningPeer(t *testing.T) string {
	t.Helper()
	for i := range 1000 {
		peer := fmt.Sprintf("podcert-peer-%d", i)
		if Hash(testItem, []string{testSelf, peer}) == peer {
			return peer
		}
	}
	t.Fatal("no candidate peer name out-hashes testSelf; widen the search")
	return ""
}

func TestAssignedToThisReplica(t *testing.T) {
	now := time.Now()
	fresh := now.Add(-time.Second)
	peer := winningPeer(t)

	tests := []struct {
		name string
		seed func(t *testing.T, h *Hasher)
		want bool
	}{
		{
			name: "no leases at all",
			seed: func(t *testing.T, h *Hasher) {},
			want: false,
		},
		{
			name: "only our own fresh lease",
			seed: func(t *testing.T, h *Hasher) {
				addLease(t, h, testSelf, ptr.To(testSelf), &fresh)
			},
			want: true,
		},
		{
			name: "fresh peer that out-hashes us wins",
			seed: func(t *testing.T, h *Hasher) {
				addLease(t, h, testSelf, ptr.To(testSelf), &fresh)
				addLease(t, h, peer, ptr.To(peer), &fresh)
			},
			want: false,
		},
		{
			name: "expired peer is ignored",
			seed: func(t *testing.T, h *Hasher) {
				addLease(t, h, testSelf, ptr.To(testSelf), &fresh)
				addLease(t, h, peer, ptr.To(peer), ptr.To(now.Add(-time.Minute)))
			},
			want: true,
		},
		{
			name: "peer with no holder or renew time is ignored without panicking",
			seed: func(t *testing.T, h *Hasher) {
				addLease(t, h, testSelf, ptr.To(testSelf), &fresh)
				addLease(t, h, "no-holder", nil, &fresh)
				addLease(t, h, "no-renew", ptr.To(peer), nil)
			},
			want: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			h := newTestHasher(t)
			tc.seed(t, h)
			if got := h.AssignedToThisReplica(context.Background(), testItem); got != tc.want {
				t.Errorf("AssignedToThisReplica(%q) = %v, want %v", testItem, got, tc.want)
			}
		})
	}
}

func TestLogUnassignedIsRateLimited(t *testing.T) {
	h := newTestHasher(t)
	addLease(t, h, testSelf, ptr.To(testSelf), ptr.To(time.Now()))
	now := time.Now()

	h.logUnassigned(context.Background(), testItem, nil, now)
	first := h.lastUnassignedLog
	h.logUnassigned(context.Background(), testItem, nil, now.Add(unassignedLogInterval/2))
	if h.lastUnassignedLog != first {
		t.Fatalf("second call within %v updated lastUnassignedLog; want it suppressed", unassignedLogInterval)
	}
	h.logUnassigned(context.Background(), testItem, nil, now.Add(unassignedLogInterval))
	if h.lastUnassignedLog == first {
		t.Fatalf("call after %v was suppressed; want it logged", unassignedLogInterval)
	}
}
