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
	reasoning  []json.RawMessage
	call       json.RawMessage
	tail       []json.RawMessage
}

func (turn trimmableTurn) items() []json.RawMessage {
	return append(append(append([]json.RawMessage{}, turn.reasoning...), turn.call), turn.tail...)
}

// Reasoning items per turn vary, so "newest 100 tool turns" cannot pass as "newest 100
// reasoning items"; prose and user messages sit between turns, so a tool turn cannot pass
// as any assistant message. Both distinctions are the point of the window.
func publishTrimmableTurns(t *testing.T, store *responsesChatReplayStore, route responsesChatReplayRoute, count int, carrier bool) ([]json.RawMessage, []trimmableTurn, map[string]carriedReplay) {
	t.Helper()
	messages := make([]json.RawMessage, 0, count*4)
	turns := make([]trimmableTurn, 0, count)
	carried := make(map[string]carriedReplay, count)
	for i := range count {
		upstreamID := fmt.Sprintf("upstream-call-%d", i)
		arguments := fmt.Sprintf(`{"turn":%d}`, i)
		outputItems := make([]json.RawMessage, 0, i%3+2)
		for j := range i%3 + 1 {
			outputItems = append(outputItems, json.RawMessage(fmt.Sprintf(
				`{"type":"reasoning","id":"rs-%d-%d","encrypted_content":"cipher-%d-%d-%s","summary":[],"content":[]}`,
				i, j, i, j, strings.Repeat("z", 64))))
		}
		callIndex := len(outputItems)
		outputItems = append(outputItems, json.RawMessage(fmt.Sprintf(
			`{"type":"function_call","call_id":%q,"name":"probe","arguments":%q,"status":"completed"}`,
			upstreamID, arguments)))
		published, err := store.Publish(responsesChatReplayPublishRequest{
			Route:            route,
			AssistantContent: json.RawMessage(`null`),
			OutputItems:      outputItems,
			Calls: []responsesChatReplayPublishCall{{
				UpstreamCallID:   upstreamID,
				Name:             "probe",
				VisibleArguments: arguments,
				OutputItemIndex:  callIndex,
			}},
		})
		if err != nil {
			t.Fatalf("Publish() error = %v", err)
		}
		projected := published.Projection.Calls[0]
		turn := trimmableTurn{proxyID: projected.ID, upstreamID: upstreamID}
		for j := range i%3 + 1 {
			turn.reasoning = append(turn.reasoning, json.RawMessage(fmt.Sprintf(
				`{"content":[],"encrypted_content":"cipher-%d-%d-%s","id":"rs-%d-%d","summary":[],"type":"reasoning"}`,
				i, j, strings.Repeat("z", 64), i, j)))
		}
		turn.call = json.RawMessage(fmt.Sprintf(
			`{"arguments":%q,"call_id":%q,"name":"probe","type":"function_call"}`, arguments, upstreamID))
		if !carrier {
			turn.reasoning = append([]json.RawMessage{}, outputItems[:callIndex]...)
			turn.call = outputItems[callIndex]
		}
		turn.tail = []json.RawMessage{
			json.RawMessage(fmt.Sprintf(`{"call_id":%q,"output":"tool-output-%d","type":"function_call_output"}`, upstreamID, i)),
			json.RawMessage(fmt.Sprintf(`{"content":[{"text":"ping-%d","type":"input_text"}],"role":"user","type":"message"}`, i)),
			json.RawMessage(fmt.Sprintf(`{"content":"note-%d","role":"assistant"}`, i)),
		}
		if carrier {
			carried[projected.ID] = carriedReplay{
				Items:            outputItems,
				Calls:            map[string]carriedCall{projected.ID: {ProxyID: projected.ID, UpstreamID: upstreamID, Name: "probe", ItemIndex: callIndex}},
				RouteDigest:      carriedRouteDigest(route),
				ProjectionDigest: carriedProjectionDigest([]byte(`""`), published.Projection.Calls),
			}
		}
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
		tool, err := json.Marshal(map[string]any{"role": "tool", "tool_call_id": projected.ID, "content": fmt.Sprintf("tool-output-%d", i)})
		if err != nil {
			t.Fatal(err)
		}
		user, err := json.Marshal(map[string]any{"role": "user", "content": fmt.Sprintf("ping-%d", i)})
		if err != nil {
			t.Fatal(err)
		}
		prose, err := json.Marshal(map[string]any{"role": "assistant", "content": fmt.Sprintf("note-%d", i)})
		if err != nil {
			t.Fatal(err)
		}
		messages = append(messages, assistant, tool, user, prose)
		turns = append(turns, turn)
	}
	return messages, turns, carried
}

// The store and the carrier are the two reasoning sources and both reach the same trim.
func trimmableTurnsUpstreamInput(t *testing.T, count int, carrier bool, options *responsesChatRequestOptions) ([]byte, []trimmableTurn) {
	t.Helper()
	store := newResponsesChatReplayStore()
	t.Cleanup(func() { _ = store.Close() })
	route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
	messages, turns, carried := publishTrimmableTurns(t, store, route, count, carrier)
	if options == nil {
		options = &responsesChatRequestOptions{}
	}
	options.UpstreamModel = "gpt-upstream"
	options.ReplayRoute = route
	if carrier {
		options.CarriedReasoning = carried
	} else {
		options.ReplayStore = store
	}
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

func forEachReasoningSource(t *testing.T, run func(t *testing.T, carrier bool)) {
	t.Helper()
	for _, source := range []struct {
		name    string
		carrier bool
	}{{"store", false}, {"carrier", true}} {
		t.Run(source.name, func(t *testing.T) { run(t, source.carrier) })
	}
}

func TestTranslateChatMessagesToResponsesDropsReasoningOlderThanRetainedToolTurns(t *testing.T) {
	forEachReasoningSource(t, func(t *testing.T, carrier bool) {
		aged := 50
		encoded, turns := trimmableTurnsUpstreamInput(t, maxReasoningToolTurns+aged, carrier, nil)
		for i, turn := range turns {
			retained := i >= aged
			for j, item := range turn.reasoning {
				if bytes.Contains(encoded, item) != retained {
					t.Fatalf("turn %d reasoning[%d] present = %v, want %v", i, j, !retained, retained)
				}
			}
		}
	})
}

func TestTranslateChatMessagesToResponsesKeepsToolCallsAtEveryAge(t *testing.T) {
	forEachReasoningSource(t, func(t *testing.T, carrier bool) {
		encoded, turns := trimmableTurnsUpstreamInput(t, maxReasoningToolTurns+50, carrier, nil)
		previous := -1
		for i, turn := range turns {
			call := bytes.Index(encoded, turn.call)
			output := bytes.Index(encoded, turn.tail[0])
			if call < 0 || output < 0 {
				t.Fatalf("turn %d call=%d output=%d, want both present", i, call, output)
			}
			if output < call || previous >= call {
				t.Fatalf("turn %d order: previous=%d call=%d output=%d", i, previous, call, output)
			}
			previous = output
		}
	})
}

// The whole upstream input, byte for byte: at the threshold the trim must splice exactly
// what it spliced before it existed, and say nothing.
func TestTranslateChatMessagesToResponsesLeavesRetainedToolTurnsByteIdentical(t *testing.T) {
	forEachReasoningSource(t, func(t *testing.T, carrier bool) {
		var sink bytes.Buffer
		options := responsesChatRequestOptions{Log: logger.NewWithWriter(logger.LevelDebug, &sink)}
		encoded, turns := trimmableTurnsUpstreamInput(t, maxReasoningToolTurns, carrier, &options)
		var expected []json.RawMessage
		for _, turn := range turns {
			expected = append(expected, turn.items()...)
		}
		want, err := json.Marshal(expected)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(encoded, want) {
			t.Fatalf("upstream input = %s\nwant %s", encoded, want)
		}
		if sink.Len() != 0 {
			t.Fatalf("threshold conversation logged %q", sink.String())
		}
	})
}

func TestTranslateChatMessagesToResponsesLogsReasoningTrimWithoutContent(t *testing.T) {
	var sink bytes.Buffer
	aged := 50
	options := responsesChatRequestOptions{Log: logger.NewWithWriter(logger.LevelDebug, &sink)}
	_, turns := trimmableTurnsUpstreamInput(t, maxReasoningToolTurns+aged, false, &options)

	var entry map[string]any
	if err := json.Unmarshal(sink.Bytes(), &entry); err != nil {
		t.Fatalf("log entry %q: %v", sink.String(), err)
	}
	dropped, droppedBytes, retained := 0, 0, 0
	for i, turn := range turns {
		if i < aged {
			dropped += len(turn.reasoning)
			for _, item := range turn.reasoning {
				droppedBytes += len(item)
			}
			continue
		}
		retained += len(turn.reasoning)
	}
	for field, want := range map[string]float64{
		"tool_turns":               float64(len(turns)),
		"aged_turns":               float64(aged),
		"retained_turns":           float64(maxReasoningToolTurns),
		"reasoning_items":          float64(dropped),
		"reasoning_bytes":          float64(droppedBytes),
		"retained_reasoning_items": float64(retained),
	} {
		if entry[field] != want {
			t.Fatalf("log %s = %v, want %v", field, entry[field], want)
		}
	}
	for _, secret := range []string{"cipher-", "tool-output-", "ping-", "note-", "probe", `\"turn\"`, turns[0].proxyID, turns[0].upstreamID} {
		if strings.Contains(sink.String(), secret) {
			t.Fatalf("log leaked %q: %s", secret, sink.String())
		}
	}
}
