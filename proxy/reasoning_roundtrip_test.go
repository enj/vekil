package proxy

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sozercan/vekil/models"
)

// The whole point, in one test: what vekil emits on turn N must be decodable
// as carried reasoning on turn N+1, with the items byte-identical. Anything
// less and Copilot receives ciphertext it cannot decrypt.
func TestCarrierSurvivesAFullTurnRoundTrip(t *testing.T) {
	outputItems := []json.RawMessage{
		json.RawMessage(`{"type":"reasoning","id":"rs_1","encrypted_content":"CIPHERTEXT","content":[],"summary":[]}`),
		json.RawMessage(`{"type":"function_call","call_id":"call_upstream_7","name":"lookup","arguments":"{}"}`),
	}

	// Turn N: vekil answers with a tool call, carrying the items.
	resp := prependCarriedReasoning(&models.AnthropicResponse{
		Content: []models.ContentBlock{
			{Type: "tool_use", ID: "call_upstream_7", Name: "lookup", Input: json.RawMessage(`{}`)},
		},
	}, outputItems)
	if len(resp.Content) != 2 || resp.Content[0].Type != "thinking" {
		t.Fatalf("carrier is not the leading block: %+v", resp.Content)
	}

	// The client persists that verbatim and replays it on turn N+1.
	replayed, err := json.Marshal(resp.Content)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	carried := extractCarriedReasoning([]models.AnthropicMessage{
		{Role: "assistant", Content: replayed},
	})

	got, ok := carried["call_upstream_7"]
	if !ok {
		t.Fatal("turn N+1 could not recover the carrier vekil emitted on turn N")
	}
	for i := range outputItems {
		if string(got[i]) != string(outputItems[i]) {
			t.Fatalf("item %d changed in transit:\n got %s\nwant %s", i, got[i], outputItems[i])
		}
	}
}

// Copilot's own call id must reach the client as the tool_use id. Minting
// proxy ids only existed to key the store; keying by the upstream id is what
// lets a carried turn be rebuilt without one.
func TestCarrierKeysOnUpstreamCallID(t *testing.T) {
	items := []json.RawMessage{json.RawMessage(`{"type":"reasoning","id":"rs_1"}`)}
	resp := prependCarriedReasoning(&models.AnthropicResponse{
		Content: []models.ContentBlock{{Type: "tool_use", ID: "call_7gCL52j7qoAMoUarPIKHL13m"}},
	}, items)
	replayed, _ := json.Marshal(resp.Content)
	carried := extractCarriedReasoning([]models.AnthropicMessage{{Role: "assistant", Content: replayed}})
	if _, ok := carried["call_7gCL52j7qoAMoUarPIKHL13m"]; !ok {
		t.Fatalf("upstream call id did not key the carrier: %v", carried)
	}
}

func TestPrependCarriedReasoningIsNoOpWithoutItems(t *testing.T) {
	original := &models.AnthropicResponse{Content: []models.ContentBlock{{Type: "text"}}}
	if got := prependCarriedReasoning(original, nil); len(got.Content) != 1 {
		t.Fatalf("added a block with nothing to carry: %+v", got.Content)
	}
	if prependCarriedReasoning(nil, []json.RawMessage{json.RawMessage(`{}`)}) != nil {
		t.Fatal("nil response should stay nil")
	}
}

// The original bug, as a test: a transcript full of call_vekil_ ids with no
// carrier and no store must keep working. Erroring here is what wedged
// sessions permanently, since those ids never leave the client transcript.
func TestLegacyReplayIDsDegradeInsteadOfWedging(t *testing.T) {
	body := []byte(`{"model":"gpt","messages":[` +
		`{"role":"assistant","tool_calls":[{"id":"call_vekil_AAAAAAAAAAAAAAAAAAAAAA","type":"function","function":{"name":"lookup","arguments":"{}"}}]},` +
		`{"role":"tool","tool_call_id":"call_vekil_AAAAAAAAAAAAAAAAAAAAAA","content":"ok"}` +
		`],"max_tokens":64}`)

	plan, err := translateChatRequestToResponses(body, responsesChatRequestOptions{UpstreamModel: "gpt"})
	if err != nil {
		t.Fatalf("legacy replay ids must degrade, not fail: %v", err)
	}
	var envelope struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(plan.Body, &envelope); err != nil {
		t.Fatal(err)
	}
	// The mandatory half must be present: a function_call_output without its
	// function_call is rejected upstream ("No tool call found for function
	// call output"), which would be a different wedge.
	var sawCall, sawOutput bool
	for _, item := range envelope.Input {
		var header struct {
			Type string `json:"type"`
		}
		_ = json.Unmarshal(item, &header)
		switch header.Type {
		case "function_call":
			sawCall = true
		case "function_call_output":
			sawOutput = true
		}
	}
	if !sawCall || !sawOutput {
		t.Fatalf("degraded turn is missing its call/output pair: call=%v output=%v", sawCall, sawOutput)
	}
}

// The carrier must reach the client through the frames a client actually
// reads. This is the bug live traffic caught: the signature was set as a field
// on content_block_start, which looks right and silently loses the payload —
// clients assemble a thinking block from its DELTAS and ignore extra fields on
// the start frame. 41 thinking blocks reached a real session with no signature,
// so every turn started without its reasoning while nothing errored.
func TestCarriedReasoningStreamsSignatureAsDelta(t *testing.T) {
	rec := httptest.NewRecorder()
	state := newAnthropicStreamState(rec, "gpt-public", "msg_test")
	if !state.start() {
		t.Fatal("stream did not start")
	}
	items := []json.RawMessage{
		json.RawMessage(`{"type":"reasoning","id":"rs_1","encrypted_content":"CIPHERTEXT"}`),
	}
	if !state.emitCarriedReasoning(items) {
		t.Fatal("emitCarriedReasoning failed")
	}

	body := rec.Body.String()
	if !strings.Contains(body, "signature_delta") {
		t.Fatalf("no signature_delta frame; the client will drop the carrier:\n%s", body)
	}
	if !strings.Contains(body, reasoningCarrierPrefix) {
		t.Fatalf("carrier payload never reached the wire:\n%s", body)
	}

	// And it must be recoverable from the delta, byte-identically — Copilot
	// decrypts what is inside.
	var signature string
	for _, line := range strings.Split(body, "\n") {
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		var evt struct {
			Delta *struct {
				Type      string `json:"type"`
				Signature string `json:"signature"`
			} `json:"delta"`
		}
		if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &evt) != nil {
			continue
		}
		if evt.Delta != nil && evt.Delta.Type == "signature_delta" {
			signature = evt.Delta.Signature
		}
	}
	if signature == "" {
		t.Fatal("signature_delta frame carried no signature")
	}
	decoded, ok := decodeReasoningCarrier(signature)
	if !ok || len(decoded) != 1 || string(decoded[0]) != string(items[0]) {
		t.Fatalf("carrier did not survive the stream: ok=%v decoded=%v", ok, decoded)
	}
}
