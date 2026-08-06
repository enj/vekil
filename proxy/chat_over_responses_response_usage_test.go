package proxy

import (
	"errors"
	"testing"
)

func TestTranslateResponsesJSONToChatAttachesUsageToPostDecodeErrors(t *testing.T) {
	tests := []struct {
		name string
		body string
		code string
	}{
		{
			name: "unexpected success error",
			body: `{"id":"resp","status":"completed","error":{"code":"invalid_prompt","message":"unexpected"},"output":[],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`,
			code: "invalid_responses_body",
		},
		{
			name: "unsupported output item",
			body: `{"id":"resp","status":"completed","output":[{"type":"computer_call"}],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`,
			code: "unsupported_responses_output",
		},
		{
			name: "unsupported terminal status",
			body: `{"id":"resp","status":"queued","output":[],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`,
			code: "unsupported_response_status",
		},
		// "replay unavailable" was removed with the store. A tool-call turn
		// with no ReplayStore used to be a hard error; the reasoning is carried
		// by the client now, so there is nothing to be unavailable and the turn
		// simply succeeds. Covered by
		// TestTranslateResponsesJSONToChatSucceedsWithoutAStore below.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := translateResponsesJSONToChat([]byte(tt.body), responsesChatResponseOptions{})
			var executionErr *chatExecutionError
			if !errors.As(err, &executionErr) {
				t.Fatalf("error = %#v, want chatExecutionError", err)
			}
			if executionErr.Code != tt.code {
				t.Fatalf("code = %q, want %q", executionErr.Code, tt.code)
			}
			if executionErr.Usage == nil || executionErr.Usage.PromptTokens != 7 || executionErr.Usage.CompletionTokens != 3 || executionErr.Usage.TotalTokens != 10 {
				t.Fatalf("usage = %#v", executionErr.Usage)
			}
		})
	}
}

// A tool-call turn with no store at all must now succeed. This is the failure
// mode the carrier removes: storage could previously be "unavailable" and take
// the turn down with it.
func TestTranslateResponsesJSONToChatSucceedsWithoutAStore(t *testing.T) {
	body := []byte(`{"id":"resp","status":"completed","output":[{"type":"function_call","call_id":"call-1","name":"lookup","arguments":"{}","status":"completed"}],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`)
	result, err := translateResponsesJSONToChat(body, responsesChatResponseOptions{})
	if err != nil {
		t.Fatalf("tool-call turn failed without a store: %v", err)
	}
	if len(result.CarriedReasoning) == 0 {
		t.Fatal("no carried reasoning; the next turn would have nothing to replay")
	}
	if result.Usage == nil || result.Usage.TotalTokens != 10 {
		t.Fatalf("usage = %#v", result.Usage)
	}
}
