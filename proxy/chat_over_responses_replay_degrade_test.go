package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sozercan/vekil/logger"
)

// degradeFixture publishes one stored turn and returns the request body that
// replays it, plus a copy whose assistant text has drifted from what was stored.
func degradeFixture(t *testing.T, store *responsesChatReplayStore, route responsesChatReplayRoute) (matching, drifted []byte, callID string) {
	t.Helper()
	published, err := store.Publish(responsesChatReplayPublishRequest{
		Route:            route,
		AssistantContent: json.RawMessage(`"checking"`),
		OutputItems: []json.RawMessage{
			json.RawMessage(`{"type":"reasoning","id":"rs_degrade","encrypted_content":"OPAQUE","content":[],"summary":[]}`),
			json.RawMessage(`{"type":"function_call","call_id":"upstream-call-1","name":"lookup","arguments":"{}","status":"completed"}`),
		},
		Calls: []responsesChatReplayPublishCall{{
			UpstreamCallID: "upstream-call-1", Name: "lookup", VisibleArguments: `{}`, OutputItemIndex: 1,
		}},
	})
	if err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	callID = published.Projection.Calls[0].ID
	matching, err = json.Marshal(map[string]any{
		"model": "gpt-public",
		"messages": []any{
			map[string]any{"role": "assistant", "content": "checking", "tool_calls": []any{map[string]any{
				"id": callID, "type": "function", "function": map[string]any{"name": "lookup", "arguments": `{}`},
			}}},
			map[string]any{"role": "tool", "tool_call_id": callID, "content": "result-1"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	drifted = []byte(strings.Replace(string(matching), `"content":"checking"`, `"content":"checking twice"`, 1))
	if bytes.Equal(drifted, matching) {
		t.Fatal("fixture no longer carries the assistant text this drift rewrites")
	}
	return matching, drifted, callID
}

// A drifted projection must reach upstream rebuilt from the visible messages,
// carrying no reasoning.
func TestProjectionMismatchDegradesToTheVisibleTranscript(t *testing.T) {
	store := newResponsesChatReplayStore()
	t.Cleanup(func() { _ = store.Close() })
	route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
	_, drifted, callID := degradeFixture(t, store, route)

	plan, err := translateChatRequestToResponses(drifted, responsesChatRequestOptions{
		UpstreamModel: "gpt-upstream", ReplayStore: store, ReplayRoute: route,
	})
	if err != nil {
		t.Fatalf("a projection mismatch must not fail the request: %v", err)
	}

	// Assert on the bytes that go on the wire: a decoded struct cannot see a field
	// omitempty dropped on the way out, and the reasoning item is exactly a field.
	input := upstreamInputJSON(t, plan)
	for _, forbidden := range []string{`"reasoning"`, "OPAQUE", "encrypted_content", "upstream-call-1"} {
		if strings.Contains(input, forbidden) {
			t.Fatalf("degraded turn carries %q: %s", forbidden, input)
		}
	}
	var items []map[string]any
	if err := json.Unmarshal([]byte(input), &items); err != nil {
		t.Fatal(err)
	}
	var call, output map[string]any
	for _, item := range items {
		switch item["type"] {
		case "function_call":
			call = item
		case "function_call_output":
			output = item
		}
	}
	if call == nil || output == nil {
		t.Fatalf("degraded turn lost its call/result pair: %s", input)
	}
	if call["call_id"] != callID || output["call_id"] != callID {
		t.Fatalf("call ids = %v / %v, want both %q: %s", call["call_id"], output["call_id"], callID, input)
	}
	if call["name"] != "lookup" || call["arguments"] != "{}" {
		t.Fatalf("degraded call = %#v, want the visible tool call", call)
	}
	if !strings.Contains(input, "checking twice") {
		t.Fatalf("degraded turn dropped the visible assistant text: %s", input)
	}
}

// TestMatchingProjectionStillRestoresStoredState pins the other half of the deal:
// where the store is live and matching, nothing about the restore changes.
func TestMatchingProjectionStillRestoresStoredState(t *testing.T) {
	store := newResponsesChatReplayStore()
	t.Cleanup(func() { _ = store.Close() })
	route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
	matching, _, _ := degradeFixture(t, store, route)

	plan, err := translateChatRequestToResponses(matching, responsesChatRequestOptions{
		UpstreamModel: "gpt-upstream", ReplayStore: store, ReplayRoute: route,
	})
	if err != nil {
		t.Fatalf("translate: %v", err)
	}
	input := upstreamInputJSON(t, plan)
	for _, want := range []string{`"encrypted_content":"OPAQUE"`, `"call_id":"upstream-call-1"`} {
		if !strings.Contains(input, want) {
			t.Fatalf("matching projection lost %s: %s", want, input)
		}
	}
}

// TestProjectionMismatchDegradeIsLogged: a degrade is a quality loss, so it must be
// distinguishable from a clean turn in the log rather than being silent.
func TestProjectionMismatchDegradeIsLogged(t *testing.T) {
	store := newResponsesChatReplayStore()
	t.Cleanup(func() { _ = store.Close() })
	route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream", RouteID: "route-a"}
	matching, drifted, _ := degradeFixture(t, store, route)

	var logs bytes.Buffer
	options := responsesChatRequestOptions{
		UpstreamModel: "gpt-upstream", ReplayStore: store, ReplayRoute: route,
		Log: logger.NewWithWriter(logger.LevelInfo, &logs),
	}
	if _, err := translateChatRequestToResponses(matching, options); err != nil {
		t.Fatalf("translate matching: %v", err)
	}
	if logs.Len() != 0 {
		t.Fatalf("a matching projection logged %q", logs.String())
	}
	if _, err := translateChatRequestToResponses(drifted, options); err != nil {
		t.Fatalf("translate drifted: %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(logs.Bytes()), &entry); err != nil {
		t.Fatalf("unmarshal %q: %v", logs.String(), err)
	}
	want := map[string]any{
		"level":    "warn",
		"msg":      "responses replay projection mismatch; continuing without reasoning continuity",
		"provider": "provider-a",
		"model":    "gpt-public",
		"route_id": "route-a",
	}
	for key, expected := range want {
		if got := entry[key]; got != expected {
			t.Fatalf("log[%s] = %#v, want %#v in %#v", key, got, expected, entry)
		}
	}
	if got, ok := entry["tool_calls"].(float64); !ok || got != 1 {
		t.Fatalf("log[tool_calls] = %#v, want 1 in %#v", entry["tool_calls"], entry)
	}
}

// Through the whole ingress: the client gets an answer, not a 400.
func TestHandleOpenAIChatCompletionsProjectionMismatchReachesUpstream(t *testing.T) {
	var upstreamBodies [][]byte
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		upstreamBodies = append(upstreamBodies, body)
		http.Error(w, "unexpected", http.StatusInternalServerError)
	}))
	defer upstream.Close()

	h := newChatExecutionTestHandler(t, upstream.URL, []string{providerEndpointResponses})
	route := responsesChatReplayRoute{ProviderID: "test-provider", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
	_, drifted, _ := degradeFixture(t, h.responsesChatReplayStore(), route)

	rec := httptest.NewRecorder()
	h.HandleOpenAIChatCompletions(rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(drifted)))

	if rec.Code == http.StatusBadRequest {
		t.Fatalf("request was rejected instead of degrading: %s", rec.Body.String())
	}
	if len(upstreamBodies) == 0 {
		t.Fatal("upstream was never called; the turn was dropped rather than degraded")
	}
	if sent := string(upstreamBodies[0]); strings.Contains(sent, "OPAQUE") || strings.Contains(sent, "upstream-call-1") {
		t.Fatalf("degraded turn carried stored state on the wire: %s", sent)
	}
}
