package proxy

import (
	"encoding/json"
	"testing"
	"time"
)

// publishOne stores a single-call group and returns the proxy call id handed
// back to the client.
func publishOne(t *testing.T, store *responsesChatReplayStore, route responsesChatReplayRoute) string {
	t.Helper()
	emptyArgs := "{}"
	published, err := store.Publish(responsesChatReplayPublishRequest{
		Route:            route,
		AssistantContent: json.RawMessage(`""`),
		OutputItems: []json.RawMessage{
			json.RawMessage(`{"type":"reasoning","encrypted_content":"OPAQUE","id":"rs_1"}`),
			json.RawMessage(`{"type":"function_call","call_id":"call_upstream_1","name":"lookup","arguments":"{}"}`),
		},
		Calls: []responsesChatReplayPublishCall{{
			UpstreamCallID:    "call_upstream_1",
			Name:              "lookup",
			VisibleArguments:  "{}",
			OriginalArguments: &emptyArgs,
			OutputItemIndex:   1,
		}},
	})
	if err != nil {
		t.Fatalf("publish: %v", err)
	}
	if len(published.Calls) != 1 {
		t.Fatalf("published calls = %d", len(published.Calls))
	}
	return published.Calls[0].ProxyCallID
}

func resolveOne(store *responsesChatReplayStore, route responsesChatReplayRoute, callID string) (responsesChatReplayResolution, error) {
	return store.Resolve(route, responsesChatReplayAssistantProjection{
		Content: json.RawMessage(`""`),
		Calls:   []responsesChatReplayProjectedCall{{ID: callID, Name: "lookup", Arguments: "{}"}},
	})
}

// TestReplayStateSurvivesRestart is the regression for the bug this file
// exists to fix: a conversation is permanently wedged when the store forgets,
// because the call ids live in the client transcript forever and re-resolve on
// every later request. A fresh store over the same directory models a proxy
// restart.
func TestReplayStateSurvivesRestart(t *testing.T) {
	dir := t.TempDir()
	route := responsesChatReplayRoute{ProviderID: "p", PublicModel: "gpt-public", UpstreamModel: "gpt-5.6-sol"}

	first := newResponsesChatReplayStoreWithOptions(responsesChatReplayStoreOptions{PersistDir: dir})
	callID := publishOne(t, first, route)
	if _, err := resolveOne(first, route, callID); err != nil {
		t.Fatalf("resolve before restart: %v", err)
	}

	// Restart: brand-new store, nothing in memory, same directory on disk.
	second := newResponsesChatReplayStoreWithOptions(responsesChatReplayStoreOptions{PersistDir: dir})
	resolution, err := resolveOne(second, route, callID)
	if err != nil {
		t.Fatalf("resolve after restart: %v (this is the wedge the fix removes)", err)
	}
	if len(resolution.OutputItems) != 2 {
		t.Fatalf("output items = %d, want the stored reasoning + function_call", len(resolution.OutputItems))
	}
	// Reasoning continuity is the point: degrading would drop this item.
	if got := string(resolution.OutputItems[0]); got == "" || !json.Valid([]byte(got)) {
		t.Fatalf("reasoning item not restored: %q", got)
	}
}

// TestReplayStateSurvivesTTLExpiry covers the other half: an idle gap longer
// than the in-memory TTL, which is what a real session hits after an hour.
func TestReplayStateSurvivesTTLExpiry(t *testing.T) {
	dir := t.TempDir()
	route := responsesChatReplayRoute{ProviderID: "p", PublicModel: "gpt-public", UpstreamModel: "gpt-5.6-sol"}
	now := time.Unix(1_700_000_000, 0)
	store := newResponsesChatReplayStoreWithOptions(responsesChatReplayStoreOptions{
		PersistDir: dir,
		TTL:        time.Hour,
		Now:        func() time.Time { return now },
	})
	callID := publishOne(t, store, route)

	now = now.Add(2 * time.Hour) // past the TTL, as in the reported failure
	if _, err := resolveOne(store, route, callID); err != nil {
		t.Fatalf("resolve after TTL expiry: %v", err)
	}
}

// TestReplayPersistDisabledKeepsMemoryOnlyBehaviour pins that an empty dir is
// still the stock behaviour, so the escape hatch genuinely disables this.
func TestReplayPersistDisabledKeepsMemoryOnlyBehaviour(t *testing.T) {
	route := responsesChatReplayRoute{ProviderID: "p", PublicModel: "gpt-public", UpstreamModel: "gpt-5.6-sol"}
	first := newResponsesChatReplayStoreWithOptions(responsesChatReplayStoreOptions{})
	callID := publishOne(t, first, route)

	second := newResponsesChatReplayStoreWithOptions(responsesChatReplayStoreOptions{})
	if _, err := resolveOne(second, route, callID); err == nil {
		t.Fatal("expected a miss with persistence disabled")
	}
}
