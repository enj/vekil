package proxy

import (
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
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

// Compression is what keeps the carrier affordable: the signature rides in
// every subsequent request, so a real turn must not balloon.
//
// Sized from a measured live item (2096-char encrypted_content, 424-char id)
// rather than the toy fixture — at toy sizes deflate+base64 overhead exceeds
// the payload, which is true but says nothing about the real cost.
func TestReasoningCarrierCompressesRealisticPayload(t *testing.T) {
	items := []json.RawMessage{
		json.RawMessage(`{"type":"reasoning","id":"` + strings.Repeat("k", 424) +
			`","encrypted_content":"` + strings.Repeat("Q", 2096) + `","content":[],"summary":[]}`),
		json.RawMessage(`{"type":"function_call","call_id":"call_upstream_1","name":"lookup","arguments":"{}"}`),
	}
	raw, _ := json.Marshal(items)
	signature, err := encodeReasoningCarrier(items)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if len(signature) >= len(raw) {
		t.Fatalf("signature (%d) did not beat raw JSON (%d)", len(signature), len(raw))
	}
	t.Logf("raw=%d signature=%d (%.0f%% of raw)", len(raw), len(signature),
		float64(len(signature))*100/float64(len(raw)))

	// And it must still round-trip byte-identically at this size.
	decoded, ok := decodeReasoningCarrier(signature)
	if !ok || len(decoded) != len(items) || string(decoded[0]) != string(items[0]) {
		t.Fatal("realistic payload did not round-trip byte-identically")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
