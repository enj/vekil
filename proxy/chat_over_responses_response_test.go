package proxy

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
)

func TestTranslateResponsesJSONToChatText(t *testing.T) {
	body, err := os.ReadFile("testdata/chat_over_responses/nonstream_text.json")
	if err != nil {
		t.Fatal(err)
	}
	result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{PublicModel: "gpt-public"})
	if err != nil {
		t.Fatalf("translateResponsesJSONToChat() error = %v", err)
	}
	if result.Response == nil || len(result.Response.Choices) != 1 {
		t.Fatalf("response = %#v", result.Response)
	}
	choice := result.Response.Choices[0]
	if choice.Message.Role != "assistant" || string(choice.Message.Content) != `"Synthetic fixture text response."` {
		t.Fatalf("message = %#v", choice.Message)
	}
	if choice.FinishReason == nil || *choice.FinishReason != "stop" {
		t.Fatalf("finish_reason = %#v", choice.FinishReason)
	}
	if result.Response.Model != "gpt-public" || result.Response.Object != "chat.completion" || result.Response.ID == "" || result.Response.Created == 0 {
		t.Fatalf("envelope = %#v", result.Response)
	}
	usage := result.Response.Usage
	if usage == nil || usage.PromptTokens != 11 || usage.CompletionTokens != 6 || usage.TotalTokens != 17 || usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CachedTokens != 3 {
		encoded, _ := json.Marshal(usage)
		t.Fatalf("usage = %s", encoded)
	}
}

func TestTranslateResponsesJSONToChatPublishesFunctionCallReplay(t *testing.T) {
	body, err := os.ReadFile("testdata/chat_over_responses/nonstream_one_tool_call.json")
	if err != nil {
		t.Fatal(err)
	}
	route := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"}
	result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{
		PublicModel: "gpt-public",
		ReplayRoute: route,
	})
	if err != nil {
		t.Fatalf("translateResponsesJSONToChat() error = %v", err)
	}
	choice := result.Response.Choices[0]
	if choice.FinishReason == nil || *choice.FinishReason != "tool_calls" || len(choice.Message.ToolCalls) != 1 {
		t.Fatalf("choice = %#v", choice)
	}
	call := choice.Message.ToolCalls[0]
	// Copilot's own call id now reaches the client verbatim. Proxy ids were
	// minted only to key the replay store; with the reasoning carried in the
	// client transcript there is no store to key, and passing the upstream id
	// through is what lets the next turn be rebuilt without one.
	if call.ID != "call_synth_lookup_001" || call.Function.Name != "lookup_synthetic_widget" || call.Function.Arguments != `{"widget":"alpha-fixture"}` {
		t.Fatalf("tool call = %#v", call)
	}
	// The turn's Responses output must come back for the client to carry.
	if len(result.CarriedReasoning) == 0 {
		t.Fatal("no carried reasoning; the next turn would have nothing to replay")
	}
	// The store is no longer consulted. What used to be published-then-resolved
	// is now carried by the client, so assert the carrier can be decoded back
	// to the same items instead.
	signature, err := encodeReasoningCarrier(result.CarriedReasoning)
	if err != nil {
		t.Fatalf("encode carrier: %v", err)
	}
	decoded, ok := decodeReasoningCarrier(signature)
	if !ok || len(decoded) != len(result.CarriedReasoning) {
		t.Fatalf("carrier did not round-trip: ok=%v decoded=%d", ok, len(decoded))
	}
}

func TestTranslateResponsesJSONToChatOutputMatrix(t *testing.T) {
	t.Run("reasoning is hidden", func(t *testing.T) {
		body, err := os.ReadFile("testdata/chat_over_responses/nonstream_reasoning_message_continuation.json")
		if err != nil {
			t.Fatal(err)
		}
		result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{PublicModel: "gpt-public"})
		if err != nil {
			t.Fatal(err)
		}
		content := string(result.Response.Choices[0].Message.Content)
		if content == "" || strings.Contains(content, "encrypted") || strings.Contains(content, "reasoning") {
			t.Fatalf("visible content = %q", content)
		}
	})

	t.Run("parallel calls use distinct proxy IDs", func(t *testing.T) {
		body, err := os.ReadFile("testdata/chat_over_responses/nonstream_parallel_tool_calls.json")
		if err != nil {
			t.Fatal(err)
		}
		result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{
			PublicModel: "gpt-public",
			ReplayRoute: responsesChatReplayRoute{ProviderID: "p", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"},
		})
		if err != nil {
			t.Fatal(err)
		}
		calls := result.Response.Choices[0].Message.ToolCalls
		// Upstream ids pass through now; they must still be distinct, since a
		// parallel group keys its carrier by each call id.
		if len(calls) != 2 || calls[0].ID == calls[1].ID ||
			strings.HasPrefix(calls[0].ID, responsesChatReplayCallIDPrefix) ||
			strings.HasPrefix(calls[1].ID, responsesChatReplayCallIDPrefix) {
			t.Fatalf("calls = %#v", calls)
		}
	})

	t.Run("incomplete token limit maps length", func(t *testing.T) {
		body, err := os.ReadFile("testdata/chat_over_responses/nonstream_incomplete_length.json")
		if err != nil {
			t.Fatal(err)
		}
		result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{PublicModel: "gpt-public"})
		if err != nil {
			t.Fatal(err)
		}
		if reason := result.Response.Choices[0].FinishReason; reason == nil || *reason != "length" {
			t.Fatalf("finish reason = %#v", reason)
		}
	})

	t.Run("unknown output item fails closed", func(t *testing.T) {
		body, err := os.ReadFile("testdata/chat_over_responses/nonstream_malformed_unknown_item.json")
		if err != nil {
			t.Fatal(err)
		}
		_, err = translateResponsesJSONToChat(body, responsesChatResponseOptions{PublicModel: "gpt-public"})
		var execErr *chatExecutionError
		if !errors.As(err, &execErr) || execErr.StatusCode != 502 || execErr.Type != "server_error" {
			t.Fatalf("error = %#v", err)
		}
	})

	t.Run("refusal text is visible", func(t *testing.T) {
		body := []byte(`{"id":"resp_refusal","created_at":1700000000,"status":"completed","output":[{"type":"message","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"Synthetic refusal."}]}]}`)
		result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{PublicModel: "gpt-public"})
		if err != nil {
			t.Fatal(err)
		}
		if string(result.Response.Choices[0].Message.Refusal) != `"Synthetic refusal."` {
			t.Fatalf("refusal = %s", result.Response.Choices[0].Message.Refusal)
		}
	})
}

func TestTranslateResponsesJSONToChatClassifiesFailedResponseAndRetainsUsage(t *testing.T) {
	body, err := os.ReadFile("testdata/chat_over_responses/nonstream_immediate_failure.json")
	if err != nil {
		t.Fatal(err)
	}
	_, err = translateResponsesJSONToChat(body, responsesChatResponseOptions{PublicModel: "gpt-public"})
	var execErr *chatExecutionError
	if !errors.As(err, &execErr) {
		t.Fatalf("error = %#v", err)
	}
	if execErr.StatusCode != 503 || execErr.Type != "server_error" || execErr.Code != "model_overloaded" || execErr.Usage == nil || execErr.Usage.PromptTokens != 5 {
		t.Fatalf("error = %+v", execErr)
	}
}

func TestTranslateResponsesJSONToChatRejectsOversizedBody(t *testing.T) {
	body := make([]byte, responsesChatMaxJSONBodyBytes+1)
	_, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{})
	var execErr *chatExecutionError
	if !errors.As(err, &execErr) || execErr.StatusCode != 502 || execErr.Code != "responses_body_too_large" {
		t.Fatalf("error = %#v", err)
	}
}

func TestTranslateResponsesJSONToChatUsageOnlyDoesNotPublishReplay(t *testing.T) {
	body, err := os.ReadFile("testdata/chat_over_responses/nonstream_one_tool_call.json")
	if err != nil {
		t.Fatal(err)
	}
	result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{
		PublicModel: "gpt-public",
		ReplayRoute: responsesChatReplayRoute{ProviderID: "provider", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"},
		UsageOnly:   true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.Usage == nil || result.Usage.PromptTokens != 20 {
		t.Fatalf("usage = %#v", result.Usage)
	}
	// (store removed; nothing recorded server-side to assert on)
}

func TestTranslateResponsesJSONToChatRejectsNonterminalToolStatusBeforePublish(t *testing.T) {
	body, err := os.ReadFile("testdata/chat_over_responses/nonstream_one_tool_call.json")
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	payload["status"] = "queued"
	body, _ = json.Marshal(payload)
	_, err = translateResponsesJSONToChat(body, responsesChatResponseOptions{
		PublicModel: "gpt-public",
		ReplayRoute: responsesChatReplayRoute{ProviderID: "provider", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"},
	})
	var executionErr *chatExecutionError
	if !errors.As(err, &executionErr) || executionErr.Code != "unsupported_response_status" {
		t.Fatalf("error = %#v", err)
	}
	// (store removed; nothing recorded server-side to assert on)
}

func TestTranslateResponsesJSONToChatDoesNotExposeIncompleteFunctionCall(t *testing.T) {
	body, err := os.ReadFile("testdata/chat_over_responses/nonstream_one_tool_call.json")
	if err != nil {
		t.Fatal(err)
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatal(err)
	}
	payload["status"] = "incomplete"
	payload["incomplete_details"] = map[string]any{"reason": "max_output_tokens"}
	payload["output"].([]any)[0].(map[string]any)["status"] = "incomplete"
	body, _ = json.Marshal(payload)
	result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{
		PublicModel: "gpt-public",
		ReplayRoute: responsesChatReplayRoute{ProviderID: "provider", PublicModel: "gpt-public", UpstreamModel: "gpt-upstream"},
	})
	if err != nil {
		t.Fatal(err)
	}
	choice := result.Response.Choices[0]
	if choice.FinishReason == nil || *choice.FinishReason != "length" || len(choice.Message.ToolCalls) != 0 {
		t.Fatalf("choice = %#v", choice)
	}
	// (store removed; nothing recorded server-side to assert on)
}

func TestTranslateResponsesJSONToChatPreservesOpaqueFunctionArguments(t *testing.T) {
	body := []byte(`{"id":"resp-opaque","created_at":1700000000,"status":"completed","output":[{"type":"function_call","id":"item","call_id":"upstream-call","name":"f","arguments":"{not-json","status":"completed"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}`)
	result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{PublicModel: "gpt", ReplayRoute: responsesChatReplayRoute{ProviderID: "p", PublicModel: "gpt", UpstreamModel: "gpt"}})
	if err != nil {
		t.Fatal(err)
	}
	call := result.Response.Choices[0].Message.ToolCalls[0]
	if call.Function.Arguments != "{not-json" {
		t.Fatalf("arguments = %q", call.Function.Arguments)
	}
	// Opaque (non-JSON) arguments must survive the carrier untouched, the same
	// obligation the store round-trip used to check.
	signature, err := encodeReasoningCarrier(result.CarriedReasoning)
	if err != nil {
		t.Fatalf("encode carrier: %v", err)
	}
	decoded, ok := decodeReasoningCarrier(signature)
	if !ok || len(decoded) != len(result.CarriedReasoning) {
		t.Fatalf("opaque arguments did not survive the carrier: ok=%v", ok)
	}
}

func TestTranslateResponsesJSONToChatAccumulatesMultipleAssistantMessages(t *testing.T) {
	body := []byte(`{"id":"resp-messages","created_at":1700000000,"status":"completed","output":[{"type":"message","id":"m1","status":"completed","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"first "}]},{"type":"message","id":"m2","status":"completed","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"second"}]}]}`)
	result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{PublicModel: "gpt"})
	if err != nil {
		t.Fatal(err)
	}
	if string(result.Response.Choices[0].Message.Content) != `"first second"` {
		t.Fatalf("content = %s", result.Response.Choices[0].Message.Content)
	}
}

func TestResponsesChatCodeOnlyFailuresAreClassified(t *testing.T) {
	tests := []struct {
		code       string
		wantStatus int
		wantType   string
	}{
		{"rate_limit_exceeded", http.StatusTooManyRequests, "rate_limit_error"},
		{"invalid_prompt", http.StatusBadRequest, "invalid_request_error"},
		{"invalid_image_url", http.StatusBadRequest, "invalid_request_error"},
		{"vector_store_timeout", http.StatusGatewayTimeout, "server_error"},
	}
	for _, tt := range tests {
		t.Run(tt.code, func(t *testing.T) {
			body := []byte(fmt.Sprintf(`{"id":"resp-error","status":"failed","error":{"code":%q,"message":"fixture"},"output":[]}`, tt.code))
			_, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{})
			var executionErr *chatExecutionError
			if !errors.As(err, &executionErr) || executionErr.StatusCode != tt.wantStatus || executionErr.Type != tt.wantType {
				t.Fatalf("error = %#v", err)
			}
		})
	}
}

func TestTranslateResponsesJSONToChatRejectsRefusalWithToolCalls(t *testing.T) {
	body := []byte(`{"id":"resp-mixed","created_at":1700000000,"status":"completed","output":[{"type":"message","id":"m","status":"completed","role":"assistant","content":[{"type":"refusal","refusal":"no"}]},{"type":"function_call","id":"f","call_id":"call","name":"tool","arguments":"{}","status":"completed"}]}`)
	_, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{ReplayRoute: responsesChatReplayRoute{ProviderID: "p", PublicModel: "gpt", UpstreamModel: "gpt"}})
	var executionErr *chatExecutionError
	if !errors.As(err, &executionErr) || executionErr.StatusCode != http.StatusBadGateway {
		t.Fatalf("error = %#v", err)
	}
	// (store removed; nothing recorded server-side to assert on)
}

func TestTranslateResponsesJSONToChatRetainsUsageWhenReplayPublishFails(t *testing.T) {
	body, err := os.ReadFile("testdata/chat_over_responses/nonstream_one_tool_call.json")
	if err != nil {
		t.Fatal(err)
	}
	// This used to force a replay-publish failure (MaxGroupBytes: 1) and assert
	// usage still rode out on the error. Nothing publishes any more -- the turn
	// is carried by the client -- so a store limit cannot fail a request at
	// all. That is the point: the failure mode is gone, not merely handled.
	//
	// The surviving obligation is that a successful turn reports usage AND
	// hands back something for the client to carry.
	result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{
		ReplayRoute: responsesChatReplayRoute{ProviderID: "p", PublicModel: "gpt", UpstreamModel: "gpt"},
	})
	if err != nil {
		t.Fatalf("a store limit must no longer be able to fail a turn: %v", err)
	}
	if result.Usage == nil || result.Usage.TotalTokens != 29 {
		t.Fatalf("usage = %#v", result.Usage)
	}
	if len(result.CarriedReasoning) == 0 {
		t.Fatal("no carried reasoning on a tool-call turn")
	}
}
