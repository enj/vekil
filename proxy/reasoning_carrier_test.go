package proxy

import (
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/sozercan/vekil/models"
)

func sampleReasoningItems() []json.RawMessage {
	return []json.RawMessage{
		json.RawMessage(`{"type":"reasoning","id":"rs_abc","encrypted_content":"OPAQUE-CIPHERTEXT","content":[],"summary":[]}`),
		json.RawMessage(`{"type":"function_call","call_id":"call_upstream_1","name":"lookup","arguments":"{}"}`),
	}
}

func TestReasoningCarrierRoundTrip(t *testing.T) {
	items := sampleReasoningItems()
	signature, err := encodeReasoningCarrier(items)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !strings.HasPrefix(signature, reasoningCarrierPrefix) {
		t.Fatalf("signature is not version-tagged: %q", signature[:min(20, len(signature))])
	}

	decoded, ok := decodeReasoningCarrier(signature)
	if !ok {
		t.Fatal("decode rejected a signature we just produced")
	}
	if len(decoded) != len(items) {
		t.Fatalf("item count = %d, want %d", len(decoded), len(items))
	}
	// Byte-identical matters: Copilot must receive its own items verbatim, and
	// encrypted_content is ciphertext it will try to decrypt.
	for i := range items {
		if string(decoded[i]) != string(items[i]) {
			t.Fatalf("item %d not byte-identical:\n got %s\nwant %s", i, decoded[i], items[i])
		}
	}
}

func TestReasoningCarrierEmptyInputProducesNoBlock(t *testing.T) {
	signature, err := encodeReasoningCarrier(nil)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if signature != "" {
		t.Fatalf("signature = %q, want empty so the caller omits the block", signature)
	}
	block, err := reasoningCarrierBlock(nil)
	if err != nil || block != nil {
		t.Fatalf("block = %v, err = %v; want no block", block, err)
	}
}

// The decoder must never fail the request. A signature arrives from a client
// and may be foreign, stale, truncated or hostile; the only sane response is
// to proceed without reasoning continuity. Rejecting instead would turn a
// recoverable quality loss into the dead conversation this design removes.
func TestReasoningCarrierRejectsGarbageWithoutErroring(t *testing.T) {
	cases := []struct {
		name      string
		signature string
	}{
		{"empty", ""},
		{"foreign prefix (real Anthropic thinking signature)", "ErUBCkYIBxgCKkDd9x2ZQ=="},
		{"our prefix, corrupt base64", reasoningCarrierPrefix + "!!!not-base64!!!"},
		{"our prefix, valid base64 but not deflate", reasoningCarrierPrefix + base64.RawURLEncoding.EncodeToString([]byte("plain"))},
		{"prefix only", reasoningCarrierPrefix},
		{"near-miss prefix", "vekil2.abc"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			items, ok := decodeReasoningCarrier(tc.signature)
			if ok || items != nil {
				t.Fatalf("decode accepted %q -> %v", tc.name, items)
			}
		})
	}
}

// A signature is attacker-reachable, so an inflate bomb is reachable too.
func TestReasoningCarrierBoundsDecompression(t *testing.T) {
	huge := make([]json.RawMessage, 0, 4096)
	filler := strings.Repeat("A", 4096)
	for i := 0; i < 4096; i++ {
		huge = append(huge, json.RawMessage(`{"type":"reasoning","encrypted_content":"`+filler+`"}`))
	}
	signature, err := encodeReasoningCarrier(huge)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	// ~16 MiB inflated from a much smaller compressed payload — past the cap.
	if _, ok := decodeReasoningCarrier(signature); ok {
		t.Fatal("decode accepted a payload past reasoningCarrierMaxDecodedBytes")
	}
}

func TestReasoningCarrierBlockShape(t *testing.T) {
	block, err := reasoningCarrierBlock(sampleReasoningItems())
	if err != nil {
		t.Fatalf("block: %v", err)
	}
	if block["type"] != "thinking" {
		t.Fatalf("type = %v, want thinking", block["type"])
	}
	// Empty thinking text is deliberate: clients already store blocks shaped
	// this way, so the carrier shows the user nothing new.
	if block["thinking"] != "" {
		t.Fatalf("thinking = %q, want empty", block["thinking"])
	}
	signature, _ := block["signature"].(string)
	if !strings.HasPrefix(signature, reasoningCarrierPrefix) {
		t.Fatalf("signature not version-tagged: %q", signature)
	}
}

// Size is a real design constraint, and compression does NOT rescue it.
//
// encrypted_content is base64 ciphertext: high entropy, incompressible.
// Measured with random bytes at the shape of a live item (1572 raw bytes ->
// ~2096 base64 chars, plus a 424-char id): raw 2685 -> signature 2763, i.e.
// 103% of raw. Deflate recovers nothing and base64url re-expands by ~3%.
//
// An earlier version of this test used strings.Repeat and "proved" 7% of raw.
// That passed for entirely the wrong reason -- repeated characters compress
// almost perfectly and are nothing like ciphertext. The honest property to
// pin is the growth RATE, because the signature rides in every subsequent
// request: ~2.1 KB per reasoning item, so a 50-turn tool session carries
// ~103 KB. That is what makes the trim policy load-bearing rather than
// optional.
func TestReasoningCarrierSizeGrowsWithCiphertext(t *testing.T) {
	item := func() json.RawMessage {
		buf := make([]byte, 1572)
		if _, err := rand.Read(buf); err != nil {
			t.Fatal(err)
		}
		return json.RawMessage(`{"type":"reasoning","encrypted_content":"` +
			base64.StdEncoding.EncodeToString(buf) + `"}`)
	}
	one, _ := encodeReasoningCarrier([]json.RawMessage{item()})
	ten := make([]json.RawMessage, 0, 10)
	for i := 0; i < 10; i++ {
		ten = append(ten, item())
	}
	many, _ := encodeReasoningCarrier(ten)

	perItem := len(many) / 10
	if perItem < 1500 || perItem > 3000 {
		t.Fatalf("per-item carrier cost = %d bytes; expected ~2.1 KB for ciphertext. "+
			"If this dropped sharply the fixture stopped being high-entropy and the "+
			"test is measuring nothing.", perItem)
	}
	if len(many) < len(one)*8 {
		t.Fatalf("10 items (%d) did not grow roughly linearly from 1 (%d); "+
			"ciphertext must not be compressing away", len(many), len(one))
	}
	t.Logf("per reasoning item ~%d bytes; 50-turn session would carry ~%d KB",
		perItem, perItem*50/1024)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func assistantWithCarrier(t *testing.T, toolUseIDs []string, items []json.RawMessage) models.AnthropicMessage {
	t.Helper()
	blocks := []map[string]any{}
	if items != nil {
		block, err := reasoningCarrierBlock(items)
		if err != nil {
			t.Fatalf("carrier block: %v", err)
		}
		blocks = append(blocks, block)
	}
	for _, id := range toolUseIDs {
		blocks = append(blocks, map[string]any{
			"type": "tool_use", "id": id, "name": "lookup", "input": map[string]any{},
		})
	}
	content, err := json.Marshal(blocks)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return models.AnthropicMessage{Role: "assistant", Content: content}
}

func TestExtractCarriedReasoningKeysEveryToolUseInTheTurn(t *testing.T) {
	items := sampleReasoningItems()
	// Parallel calls in one assistant turn share the turn's output array.
	msgs := []models.AnthropicMessage{
		assistantWithCarrier(t, []string{"call_a", "call_b"}, items),
	}
	carried := extractCarriedReasoning(msgs)
	if len(carried) != 2 {
		t.Fatalf("carried %d ids, want 2", len(carried))
	}
	for _, id := range []string{"call_a", "call_b"} {
		got, ok := carried[id]
		if !ok || string(got[0]) != string(items[0]) {
			t.Fatalf("id %q did not map to the turn's items", id)
		}
	}
}

func TestExtractCarriedReasoningIsPerTurn(t *testing.T) {
	first := []json.RawMessage{json.RawMessage(`{"type":"reasoning","id":"turn1"}`)}
	second := []json.RawMessage{json.RawMessage(`{"type":"reasoning","id":"turn2"}`)}
	msgs := []models.AnthropicMessage{
		assistantWithCarrier(t, []string{"call_1"}, first),
		assistantWithCarrier(t, []string{"call_2"}, second),
	}
	carried := extractCarriedReasoning(msgs)
	if string(carried["call_1"][0]) != string(first[0]) {
		t.Fatalf("call_1 got the wrong turn: %s", carried["call_1"][0])
	}
	if string(carried["call_2"][0]) != string(second[0]) {
		t.Fatalf("call_2 got the wrong turn: %s", carried["call_2"][0])
	}
}

// A transcript recorded before this existed, or by a client that drops
// thinking blocks, must produce no carrier rather than an error. That absence
// is what lets the caller degrade instead of wedging the conversation.
func TestExtractCarriedReasoningToleratesTranscriptsWithoutCarriers(t *testing.T) {
	cases := []struct {
		name string
		msgs []models.AnthropicMessage
	}{
		{"tool_use with no thinking block", []models.AnthropicMessage{
			assistantWithCarrier(t, []string{"call_legacy"}, nil)}},
		{"string content, no blocks", []models.AnthropicMessage{
			{Role: "assistant", Content: json.RawMessage(`"plain text"`)}}},
		{"user role is never a carrier", []models.AnthropicMessage{
			{Role: "user", Content: json.RawMessage(`[{"type":"text","text":"hi"}]`)}}},
		{"foreign thinking signature", []models.AnthropicMessage{
			{Role: "assistant", Content: json.RawMessage(
				`[{"type":"thinking","thinking":"","signature":"ErUBCkYIBxgC"},` +
					`{"type":"tool_use","id":"call_x","name":"f","input":{}}]`)}}},
		{"empty", nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if carried := extractCarriedReasoning(tc.msgs); carried != nil {
				t.Fatalf("expected no carrier, got %v", carried)
			}
		})
	}
}

// A carrier with no tool_use in its message has nothing to key on. Dropping it
// is correct: reasoning is only replayed to accompany a tool result.
func TestExtractCarriedReasoningIgnoresCarrierWithoutToolUse(t *testing.T) {
	msgs := []models.AnthropicMessage{assistantWithCarrier(t, nil, sampleReasoningItems())}
	if carried := extractCarriedReasoning(msgs); carried != nil {
		t.Fatalf("expected no carrier without tool_use, got %v", carried)
	}
}
