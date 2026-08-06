package proxy

import (
	"encoding/json"
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
