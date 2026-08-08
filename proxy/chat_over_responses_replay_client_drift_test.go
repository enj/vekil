package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/sozercan/vekil/logger"
	"github.com/sozercan/vekil/models"
)

// clientDriftFixture publishes one turn, hands back the carrier the client would hold,
// and a Chat body replaying that turn with `returned` as the call's arguments.
func clientDriftFixture(t *testing.T, store *responsesChatReplayStore, route responsesChatReplayRoute, name, emitted, returned string) (map[string]carriedReplay, []byte, string) {
	t.Helper()
	items := []json.RawMessage{
		json.RawMessage(`{"type":"reasoning","id":"rs_drift","encrypted_content":"OPAQUE","content":[],"summary":[]}`),
		responsesFunctionCallItem("upstream-call-1", name, emitted),
	}
	published, err := store.Publish(responsesChatReplayPublishRequest{
		Route:            route,
		AssistantContent: json.RawMessage(`"checking"`),
		OutputItems:      items,
		Calls: []responsesChatReplayPublishCall{{
			UpstreamCallID: "upstream-call-1", Name: name, VisibleArguments: emitted, OutputItemIndex: 1,
		}},
	})
	if err != nil {
		t.Fatalf("Publish() error = %v", err)
	}
	callID := published.Projection.Calls[0].ID

	resp := prependCarriedReasoning(&models.AnthropicResponse{
		Content: []models.ContentBlock{{Type: "tool_use", ID: callID, Name: name, Input: json.RawMessage(emitted)}},
	}, carriedTurnFromPublished(route, items, published))
	blocks, err := json.Marshal(resp.Content)
	if err != nil {
		t.Fatal(err)
	}
	carried := extractCarriedReasoning([]models.AnthropicMessage{{Role: "assistant", Content: blocks}})
	if _, ok := carried[callID]; !ok {
		t.Fatalf("fixture emitted no usable carrier for %s", callID)
	}

	body, err := json.Marshal(map[string]any{
		"model": route.PublicModel,
		"messages": []any{
			map[string]any{"role": "assistant", "content": "checking", "tool_calls": []any{map[string]any{
				"id": callID, "type": "function", "function": map[string]any{"name": name, "arguments": returned},
			}}},
			map[string]any{"role": "tool", "tool_call_id": callID, "content": "result-1"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return carried, body, callID
}

// requireStoreRejectsArguments keeps the fixture honest: if the store ever starts
// accepting the client's rewrite, these tests must fail rather than pass vacuously.
func requireStoreRejectsArguments(t *testing.T, store *responsesChatReplayStore, route responsesChatReplayRoute, callID, name, returned string) {
	t.Helper()
	_, err := resolveResponsesChatReplay(store, route, responsesChatReplayAssistantProjection{
		Content: json.RawMessage(`"checking"`),
		Calls:   []responsesChatReplayProjectedCall{{ID: callID, Name: name, Arguments: returned}},
	})
	if !errors.Is(err, &responsesChatReplayProjectionError{}) {
		t.Fatalf("store accepted the drifted arguments (err = %v); fixture no longer reproduces the drift", err)
	}
}

// Claude Code rewrites tool_use.input between turns and returns the rewrite. Every case
// below is taken from a live gpt-5.6-sol session; the store binds arguments and rejects
// all of them, so reasoning continuity has to come from the carrier, which does not.
func TestClientRewrittenArgumentsStillRestoreCarriedReasoning(t *testing.T) {
	for _, testCase := range []struct {
		name     string
		tool     string
		emitted  string
		returned string
	}{{
		name:     "materialised schema default",
		tool:     "Edit",
		emitted:  `{"file_path":"/tmp/a","new_string":"b","old_string":"a"}`,
		returned: `{"file_path":"/tmp/a","new_string":"b","old_string":"a","replace_all":false}`,
	}, {
		name:     "client-invented alias keys",
		tool:     "SendMessage",
		emitted:  `{"message":"go","summary":"s","to":"agent"}`,
		returned: `{"content":"go","message":"go","recipient":"agent","summary":"s","to":"agent","type":"message"}`,
	}, {
		name:     "field the client generated after the call",
		tool:     "ExitPlanMode",
		emitted:  `{"allowedPrompts":[{"prompt":"build","tool":"Bash"}]}`,
		returned: `{"allowedPrompts":[{"prompt":"build","tool":"Bash"}],"plan":"# Plan","planFilePath":"/Users/mo/.claude/plans/x.md"}`,
	}, {
		name:     "text the client edited",
		tool:     "Write",
		emitted:  `{"content":"line \n","file_path":"/tmp/p.patch"}`,
		returned: `{"content":"line\n","file_path":"/tmp/p.patch"}`,
	}} {
		t.Run(testCase.name, func(t *testing.T) {
			store := newResponsesChatReplayStore()
			t.Cleanup(func() { _ = store.Close() })
			route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
			carried, body, callID := clientDriftFixture(t, store, route, testCase.tool, testCase.emitted, testCase.returned)
			requireStoreRejectsArguments(t, store, route, callID, testCase.tool, testCase.returned)

			plan, err := translateChatRequestToResponses(body, responsesChatRequestOptions{
				UpstreamModel: "gpt-upstream", ReplayStore: store, ReplayRoute: route, CarriedReasoning: carried,
			})
			if err != nil {
				t.Fatalf("translate: %v", err)
			}
			// Assert on the wire bytes: a decoded item cannot show an omitempty field
			// dropped on the way out, and the ciphertext is exactly such a field.
			input := upstreamInputJSON(t, plan)
			if !strings.Contains(input, `"encrypted_content":"OPAQUE"`) {
				t.Fatalf("client argument drift threw away reasoning continuity: %s", input)
			}
			if !strings.Contains(input, `"call_id":"upstream-call-1"`) {
				t.Fatalf("restored turn lost its upstream call binding: %s", input)
			}
			// The client's arguments are what upstream must see, not the stored ones.
			if !strings.Contains(input, jsonStringOf(t, testCase.returned)) {
				t.Fatalf("restored turn did not forward the client's arguments: %s", input)
			}
		})
	}
}

func jsonStringOf(t *testing.T, value string) string {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

// A degrade that survives the carrier must say which side of the projection moved, in
// hashes: the projections themselves are prompt data and must never reach a log.
func TestDegradeLogNamesTheDivergingSideInHashes(t *testing.T) {
	store := newResponsesChatReplayStore()
	t.Cleanup(func() { _ = store.Close() })
	route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
	emitted := `{"file_path":"/tmp/a","new_string":"b","old_string":"a"}`
	returned := `{"file_path":"/tmp/a","new_string":"b","old_string":"a","replace_all":false}`
	carried, body, callID := clientDriftFixture(t, store, route, "Edit", emitted, returned)
	requireStoreRejectsArguments(t, store, route, callID, "Edit", returned)

	// A carrier minted under a different route cannot restore, so the turn degrades even
	// though its content and call sequence are provably unchanged.
	var logs bytes.Buffer
	elsewhere := route
	elsewhere.UpstreamModel = "gpt-other"
	if _, err := translateChatRequestToResponses(body, responsesChatRequestOptions{
		UpstreamModel: "gpt-upstream", ReplayStore: store, ReplayRoute: route,
		CarriedReasoning: reroutedCarrier(t, carried, elsewhere),
		Log:              logger.NewWithWriter(logger.LevelInfo, &logs),
	}); err != nil {
		t.Fatalf("translate: %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(logs.Bytes()), &entry); err != nil {
		t.Fatalf("unmarshal %q: %v", logs.String(), err)
	}
	if entry["diverged"] != "arguments" {
		t.Fatalf("log[diverged] = %#v, want %q in %#v", entry["diverged"], "arguments", entry)
	}
	projection, _ := entry["projection"].(string)
	if projection == "" || entry["carried_projection"] != projection {
		t.Fatalf("matching projections must log as equal hashes: %#v", entry)
	}
	for _, leaked := range []string{"checking", "old_string", "/tmp/a"} {
		if strings.Contains(logs.String(), leaked) {
			t.Fatalf("degrade log leaked prompt data %q: %s", leaked, logs.String())
		}
	}
}

func reroutedCarrier(t *testing.T, carried map[string]carriedReplay, route responsesChatReplayRoute) map[string]carriedReplay {
	t.Helper()
	rerouted := make(map[string]carriedReplay, len(carried))
	for id, replay := range carried {
		replay.RouteDigest = carriedRouteDigest(route)
		rerouted[id] = replay
	}
	return rerouted
}

// Without a carrier there is nothing to narrow the mismatch with, and saying so beats
// naming a side on no evidence.
func TestDegradeLogSaysUnknownWithoutACarrier(t *testing.T) {
	store := newResponsesChatReplayStore()
	t.Cleanup(func() { _ = store.Close() })
	route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
	_, drifted, _ := degradeFixture(t, store, route)

	var logs bytes.Buffer
	if _, err := translateChatRequestToResponses(drifted, responsesChatRequestOptions{
		UpstreamModel: "gpt-upstream", ReplayStore: store, ReplayRoute: route,
		Log: logger.NewWithWriter(logger.LevelInfo, &logs),
	}); err != nil {
		t.Fatalf("translate: %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(logs.Bytes()), &entry); err != nil {
		t.Fatalf("unmarshal %q: %v", logs.String(), err)
	}
	if entry["diverged"] != "unknown" || entry["carried_projection"] != "" {
		t.Fatalf("carrier-less degrade must report unknown: %#v", entry)
	}
	if projection, _ := entry["projection"].(string); projection == "" {
		t.Fatalf("degrade must still log the recomputed projection hash: %#v", entry)
	}
}
