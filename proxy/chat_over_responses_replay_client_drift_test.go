package proxy

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

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
