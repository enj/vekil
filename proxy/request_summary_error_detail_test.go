package proxy

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// Upstream error prose quotes the offending request value back, so it must not
// reach the log even though the enumerated type, code and param may.
func TestUpstreamAuthoredErrorMessageStaysOutOfTheLog(t *testing.T) {
	const secret = "SSN 123-45-6789 and the user's prompt"
	upstreamBody := []byte(`{"error":{"type":"invalid_request_error","code":"bad_value","param":"messages","message":"Invalid value for 'messages[0].content': ` + secret + `"}}`)

	for _, tc := range []struct {
		name string
		err  error
		want map[string]string
	}{
		{
			name: "upstream responses error keeps its classifiers but loses the prose",
			err:  responsesChatExecutionErrorFromUpstream(&upstreamError{statusCode: 400, body: upstreamBody}),
			want: map[string]string{"error_type": "invalid_request_error", "error_code": "bad_value", "error_param": "messages"},
		},
		{
			name: "upstream stream termination loses the prose",
			err:  chatExecutionErrorFromStreamTermination(errors.New("upstream said " + secret)),
			want: map[string]string{"error_type": "server_error", "error_code": "responses_stream_failed"},
		},
		{
			name: "vekil's own diagnosis is logged in full",
			err:  replayChatExecutionError(responsesChatReplayProjectionCode, responsesChatReplayProjectionMessage),
			want: map[string]string{
				"error_type":    "invalid_request_error",
				"error_code":    responsesChatReplayProjectionCode,
				"error_param":   "messages",
				"error_message": responsesChatReplayProjectionMessage,
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, summary := WithRequestSummary(context.Background())
			var executionErr *chatExecutionError
			if !errors.As(tc.err, &executionErr) {
				t.Fatalf("err = %T, want *chatExecutionError", tc.err)
			}
			observeChatExecutionError(ctx, executionErr)

			got := map[string]string{}
			for _, field := range summary.LoggerFields() {
				if !strings.HasPrefix(field.Key, "error_") {
					continue
				}
				value, ok := field.Value.(string)
				if !ok {
					t.Fatalf("%s = %#v, want a string", field.Key, field.Value)
				}
				if strings.Contains(value, secret) {
					t.Fatalf("%s leaked request content: %q", field.Key, value)
				}
				got[field.Key] = value
			}
			if len(got) != len(tc.want) {
				t.Fatalf("fields = %#v, want %#v", got, tc.want)
			}
			for key, want := range tc.want {
				if got[key] != want {
					t.Fatalf("fields[%s] = %q, want %q", key, got[key], want)
				}
			}
		})
	}
}
