package proxy

import (
	"encoding/json"
	"testing"
)

func TestTranslateChatRequestToResponsesNormalizesReplayAssistantContentParts(t *testing.T) {
	// Carried by the client instead of published to a store; the tool id is
	// Copilot's own, which is what a carried turn keys on.
	carriedItems := []json.RawMessage{
		json.RawMessage(`{"type":"message","role":"assistant","content":[{"type":"output_text","text":"checking "},{"type":"output_text","text":"status"}]}`),
		json.RawMessage(`{"type":"function_call","call_id":"upstream-call-1","name":"lookup","arguments":"{}","status":"completed"}`),
	}
	call := struct{ ID, Name, Arguments string }{ID: "upstream-call-1", Name: "lookup", Arguments: `{}`}
	request := map[string]any{
		"model": "gpt-public",
		"messages": []any{
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "text", "text": "checking "},
					map[string]any{"type": "text", "text": "status"},
				},
				"tool_calls": []any{map[string]any{
					"id":   call.ID,
					"type": "function",
					"function": map[string]any{
						"name":      call.Name,
						"arguments": call.Arguments,
					},
				}},
			},
			map[string]any{"role": "tool", "tool_call_id": call.ID, "content": "done"},
		},
	}
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}

	plan, err := translateChatRequestToResponses(body, responsesChatRequestOptions{
		UpstreamModel:    "gpt-upstream",
		CarriedReasoning: map[string][]json.RawMessage{"upstream-call-1": carriedItems},
	})
	if err != nil {
		t.Fatalf("translateChatRequestToResponses() error = %v", err)
	}
	var upstream struct {
		Input []map[string]any `json:"input"`
	}
	if err := json.Unmarshal(plan.Body, &upstream); err != nil {
		t.Fatal(err)
	}
	if len(upstream.Input) != 3 {
		t.Fatalf("upstream input = %#v", upstream.Input)
	}
	if upstream.Input[0]["type"] != "message" || upstream.Input[1]["type"] != "function_call" || upstream.Input[2]["type"] != "function_call_output" {
		t.Fatalf("upstream input = %#v", upstream.Input)
	}
}

// TestTranslateChatRequestToResponsesPreservesReplayNullEmptyContentCompatibility
// is gone with the store.
//
// It published an assistant turn with null vs "" content and asserted both
// resolved back identically -- a projection-matching concern that only existed
// because the store had to re-match what the client echoed. The carrier hands
// the items back directly, so there is no projection to match and nothing to
// diverge.
