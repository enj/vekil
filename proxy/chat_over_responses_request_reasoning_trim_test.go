package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/sozercan/vekil/logger"
)

type trimmableTurn struct {
	proxyID    string
	upstreamID string
	reasoning  json.RawMessage
	call       json.RawMessage
	output     string
}

// One reasoning item and one call per turn, so a dropped item is one turn's worth.
func publishTrimmableTurns(t *testing.T, store *responsesChatReplayStore, route responsesChatReplayRoute, count int) ([]json.RawMessage, []trimmableTurn) {
	t.Helper()
	messages := make([]json.RawMessage, 0, count*2)
	turns := make([]trimmableTurn, 0, count)
	for i := range count {
		upstreamID := fmt.Sprintf("upstream-call-%d", i)
		reasoning := json.RawMessage(fmt.Sprintf(
			`{"type":"reasoning","id":"rs-%d","encrypted_content":"cipher-%d-%s","summary":[],"content":[]}`,
			i, i, strings.Repeat("z", 64)))
		call := json.RawMessage(fmt.Sprintf(
			`{"type":"function_call","call_id":%q,"name":"probe","arguments":"{\"turn\":%d}","status":"completed"}`,
			upstreamID, i))
		published, err := store.Publish(responsesChatReplayPublishRequest{
			Route:            route,
			AssistantContent: json.RawMessage(`null`),
			OutputItems:      []json.RawMessage{reasoning, call},
			Calls: []responsesChatReplayPublishCall{{
				UpstreamCallID:   upstreamID,
				Name:             "probe",
				VisibleArguments: fmt.Sprintf(`{"turn":%d}`, i),
				OutputItemIndex:  1,
			}},
		})
		if err != nil {
			t.Fatalf("Publish() error = %v", err)
		}
		projected := published.Projection.Calls[0]
		output := fmt.Sprintf("tool-output-%d", i)
		assistant, err := json.Marshal(map[string]any{
			"role":    "assistant",
			"content": nil,
			"tool_calls": []any{map[string]any{
				"id":       projected.ID,
				"type":     "function",
				"function": map[string]any{"name": projected.Name, "arguments": projected.Arguments},
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
		tool, err := json.Marshal(map[string]any{"role": "tool", "tool_call_id": projected.ID, "content": output})
		if err != nil {
			t.Fatal(err)
		}
		messages = append(messages, assistant, tool)
		turns = append(turns, trimmableTurn{
			proxyID: projected.ID, upstreamID: upstreamID, reasoning: reasoning, call: call, output: output,
		})
	}
	return messages, turns
}

func trimmableTurnsUpstreamInput(t *testing.T, count int, options *responsesChatRequestOptions) ([]byte, []trimmableTurn) {
	t.Helper()
	store := newResponsesChatReplayStore()
	t.Cleanup(func() { _ = store.Close() })
	route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
	messages, turns := publishTrimmableTurns(t, store, route, count)
	if options == nil {
		options = &responsesChatRequestOptions{}
	}
	options.UpstreamModel = "gpt-upstream"
	options.ReplayStore = store
	options.ReplayRoute = route
	input, err := translateChatMessagesToResponses(messages, *options)
	if err != nil {
		t.Fatalf("translateChatMessagesToResponses() error = %v", err)
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	return encoded, turns
}

func TestTranslateChatMessagesToResponsesDropsReasoningOlderThanRetainedToolTurns(t *testing.T) {
	aged := 50
	encoded, turns := trimmableTurnsUpstreamInput(t, maxReasoningToolTurns+aged, nil)
	for i, turn := range turns {
		retained := i >= aged
		if bytes.Contains(encoded, turn.reasoning) != retained {
			t.Fatalf("turn %d reasoning present = %v, want %v", i, !retained, retained)
		}
	}
}

func TestTranslateChatMessagesToResponsesKeepsToolCallsAtEveryAge(t *testing.T) {
	encoded, turns := trimmableTurnsUpstreamInput(t, maxReasoningToolTurns+50, nil)
	previous := -1
	for i, turn := range turns {
		call := bytes.Index(encoded, turn.call)
		if call < 0 {
			t.Fatalf("turn %d function_call missing from upstream input", i)
		}
		wantOutput, err := json.Marshal(map[string]any{
			"type": "function_call_output", "call_id": turn.upstreamID, "output": turn.output,
		})
		if err != nil {
			t.Fatal(err)
		}
		output := bytes.Index(encoded, wantOutput)
		if output < call || previous >= call {
			t.Fatalf("turn %d order: previous=%d call=%d output=%d", i, previous, call, output)
		}
		previous = output
	}
}

// The whole upstream input, byte for byte: at the threshold the trim must splice exactly
// what it spliced before it existed.
func TestTranslateChatMessagesToResponsesLeavesRetainedToolTurnsByteIdentical(t *testing.T) {
	encoded, turns := trimmableTurnsUpstreamInput(t, maxReasoningToolTurns, nil)
	expected := make([]json.RawMessage, 0, len(turns)*3)
	for _, turn := range turns {
		output, err := json.Marshal(map[string]any{
			"type": "function_call_output", "call_id": turn.upstreamID, "output": turn.output,
		})
		if err != nil {
			t.Fatal(err)
		}
		expected = append(expected, turn.reasoning, turn.call, output)
	}
	want, err := json.Marshal(expected)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(encoded, want) {
		t.Fatalf("upstream input = %s\nwant %s", encoded, want)
	}
}

func TestTranslateChatMessagesToResponsesLogsReasoningTrimWithoutContent(t *testing.T) {
	var sink bytes.Buffer
	aged := 50
	options := responsesChatRequestOptions{Log: logger.NewWithWriter(logger.LevelDebug, &sink)}
	_, turns := trimmableTurnsUpstreamInput(t, maxReasoningToolTurns+aged, &options)

	var entry map[string]any
	if err := json.Unmarshal(sink.Bytes(), &entry); err != nil {
		t.Fatalf("log entry %q: %v", sink.String(), err)
	}
	droppedBytes := 0
	for _, turn := range turns[:aged] {
		droppedBytes += len(turn.reasoning)
	}
	for field, want := range map[string]float64{
		"tool_turns":      float64(len(turns)),
		"trimmed_turns":   float64(aged),
		"retained_turns":  float64(maxReasoningToolTurns),
		"reasoning_items": float64(aged),
		"reasoning_bytes": float64(droppedBytes),
	} {
		if entry[field] != want {
			t.Fatalf("log %s = %v, want %v", field, entry[field], want)
		}
	}
	for _, secret := range []string{"cipher-", "tool-output-", "probe", `\"turn\"`, turns[0].proxyID, turns[0].upstreamID} {
		if strings.Contains(sink.String(), secret) {
			t.Fatalf("log leaked %q: %s", secret, sink.String())
		}
	}
}
