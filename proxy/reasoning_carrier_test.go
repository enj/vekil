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
	signature, err := encodeReasoningCarrier(carriedTurn{Items: items})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if !strings.HasPrefix(signature, reasoningCarrierPrefix) {
		t.Fatalf("signature is not version-tagged: %q", signature[:min(20, len(signature))])
	}

	replay, ok := decodeReasoningCarrier(signature, nil)
	if !ok {
		t.Fatal("decode rejected a signature we just produced")
	}
	if len(replay.Items) != len(items) {
		t.Fatalf("item count = %d, want %d", len(replay.Items), len(items))
	}
	// encrypted_content is ciphertext Copilot will try to decrypt.
	for i := range items {
		if string(replay.Items[i]) != string(items[i]) {
			t.Fatalf("item %d not byte-identical:\n got %s\nwant %s", i, replay.Items[i], items[i])
		}
	}
}

func TestReasoningCarrierEmptyInputProducesNoBlock(t *testing.T) {
	signature, err := encodeReasoningCarrier(carriedTurn{})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if signature != "" {
		t.Fatalf("signature = %q, want empty so the caller omits the block", signature)
	}
	block, err := reasoningCarrierBlock(carriedTurn{})
	if err != nil || block != nil {
		t.Fatalf("block = %v, err = %v; want no block", block, err)
	}
}

// The decoder must never fail the request: a signature may be foreign, stale,
// truncated or hostile, and rejecting would recreate the wedge.
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
			replay, ok := decodeReasoningCarrier(tc.signature, nil)
			if ok || replay.Items != nil {
				t.Fatalf("decode accepted %q -> %v", tc.name, replay.Items)
			}
		})
	}
}

// A signature is attacker-reachable, so an inflate bomb is too. Over-cap carriers
// are dropped, which after a restart is the wedge again: pin the cap itself.
func TestReasoningCarrierBoundsDecompression(t *testing.T) {
	// Random filler, so flate cannot shrink it into a different test.
	carrierOfDecodedSize := func(t *testing.T, delta int) string {
		t.Helper()
		base, err := encodeReasoningCarrier(carriedTurn{Items: []json.RawMessage{json.RawMessage(`{"encrypted_content":""}`)}})
		if err != nil || base == "" {
			t.Fatalf("encode: %v", err)
		}
		overhead := len(`{"items":[{"encrypted_content":""}]}`) + len(`,"route_digest":""`)
		filler := make([]byte, reasoningCarrierMaxDecodedBytes+delta-overhead)
		if _, err := rand.Read(filler); err != nil {
			t.Fatal(err)
		}
		item, err := json.Marshal(map[string]string{"encrypted_content": base64.RawStdEncoding.EncodeToString(filler)[:len(filler)]})
		if err != nil {
			t.Fatal(err)
		}
		signature, err := encodeReasoningCarrier(carriedTurn{Items: []json.RawMessage{item}})
		if err != nil {
			t.Fatalf("encode: %v", err)
		}
		return signature
	}

	// Absolute, so moving the cap cannot pass by moving the boundary with it.
	if reasoningCarrierMaxDecodedBytes < 300*2100 || reasoningCarrierMaxDecodedBytes > 1<<20 {
		t.Fatalf("cap = %d bytes, outside the range the carrier is sized for", reasoningCarrierMaxDecodedBytes)
	}
	if _, ok := decodeReasoningCarrier(carrierOfDecodedSize(t, -256), nil); !ok {
		t.Fatal("decode rejected a carrier just under the cap; a long real turn would wedge")
	}
	if _, ok := decodeReasoningCarrier(carrierOfDecodedSize(t, 256), nil); ok {
		t.Fatal("decode accepted a payload past reasoningCarrierMaxDecodedBytes")
	}
}

func TestReasoningCarrierBlockShape(t *testing.T) {
	block, err := reasoningCarrierBlock(carriedTurn{Items: sampleReasoningItems()})
	if err != nil {
		t.Fatalf("block: %v", err)
	}
	if block.Type != "thinking" {
		t.Fatalf("type = %v, want thinking", block.Type)
	}
	// Empty thinking text is deliberate: the carrier shows the user nothing new.
	if block.Thinking == nil || *block.Thinking != "" {
		t.Fatalf("thinking = %v, want a present empty string", block.Thinking)
	}
	if !strings.HasPrefix(block.Signature, reasoningCarrierPrefix) {
		t.Fatalf("signature not version-tagged: %q", block.Signature)
	}
}

// Compression does not rescue size -- encrypted_content is ciphertext -- and the
// signature rides in every later request, so pin the growth RATE, not a total.
func TestReasoningCarrierSizeGrowsWithCiphertext(t *testing.T) {
	item := func() json.RawMessage {
		buf := make([]byte, 1572)
		if _, err := rand.Read(buf); err != nil {
			t.Fatal(err)
		}
		return json.RawMessage(`{"type":"reasoning","encrypted_content":"` +
			base64.StdEncoding.EncodeToString(buf) + `"}`)
	}
	one, _ := encodeReasoningCarrier(carriedTurn{Items: []json.RawMessage{item()}})
	ten := make([]json.RawMessage, 0, 10)
	for i := 0; i < 10; i++ {
		ten = append(ten, item())
	}
	many, _ := encodeReasoningCarrier(carriedTurn{Items: ten})

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

func assistantWithCarrier(t *testing.T, toolUseIDs []string, items []json.RawMessage) models.AnthropicMessage {
	t.Helper()
	blocks := []any{}
	if items != nil {
		block, err := reasoningCarrierBlock(carriedTurn{Items: items})
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
	msgs := []models.AnthropicMessage{
		assistantWithCarrier(t, []string{"call_a", "call_b"}, items),
	}
	carried := extractCarriedReasoning(msgs)
	if len(carried) != 2 {
		t.Fatalf("carried %d ids, want 2", len(carried))
	}
	for _, id := range []string{"call_a", "call_b"} {
		got, ok := carried[id]
		if !ok || string(got.Items[0]) != string(items[0]) {
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
	if string(carried["call_1"].Items[0]) != string(first[0]) {
		t.Fatalf("call_1 got the wrong turn: %s", carried["call_1"].Items[0])
	}
	if string(carried["call_2"].Items[0]) != string(second[0]) {
		t.Fatalf("call_2 got the wrong turn: %s", carried["call_2"].Items[0])
	}
}

// Absence, not an error: that is what lets the caller fall through to the store.
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

// A bare thinking block never replayed a turn, so its items must not escape.
// Paired with a real turn, or the "no carrier" early return masks it.
func TestExtractCarriedReasoningIgnoresCarrierWithoutToolUse(t *testing.T) {
	block := func(t *testing.T, id string) *models.ContentBlock {
		t.Helper()
		b, err := reasoningCarrierBlock(carriedTurn{
			Items: []json.RawMessage{json.RawMessage(`{"type":"reasoning","id":"` + id + `"}`)},
		})
		if err != nil {
			t.Fatalf("carrier block: %v", err)
		}
		return b
	}
	bare, err := json.Marshal([]any{block(t, "spliced")})
	if err != nil {
		t.Fatal(err)
	}
	turn, err := json.Marshal([]any{block(t, "turn"), toolUseBlock(mintedCallID(t))})
	if err != nil {
		t.Fatal(err)
	}

	carried := extractCarriedReasoning([]models.AnthropicMessage{
		{Role: "assistant", Content: turn},
		{Role: "assistant", Content: bare},
	})
	if len(carried) != 1 {
		t.Fatalf("carried %d ids, want only the tool-call turn's", len(carried))
	}
	for id, replay := range carried {
		if !strings.Contains(string(replay.Items[0]), `"turn"`) {
			t.Fatalf("id %q got the bare block's items: %s", id, replay.Items[0])
		}
	}
}

// The digest binds a carrier to the route that minted it: Copilot's ciphertext is
// model-bound, and it is also how a continuation is matched back to its own tier.
func TestCarrierRouteDigestRejectsAnotherRoute(t *testing.T) {
	minted := responsesChatReplayRoute{ProviderID: "provider-a", PublicModel: "gpt-public", UpstreamModel: "gpt-5.6-sol", RouteID: "sol-route", PolicyTier: "powerful"}
	other := minted
	other.RouteID, other.UpstreamModel, other.PolicyTier = "luna-route", "gpt-5.6-luna", "lightweight"

	projected := []responsesChatReplayProjectedCall{{ID: "call_vekil_x", Name: "lookup", Arguments: "{}"}}
	content := json.RawMessage(`"checking"`)
	signature, err := encodeReasoningCarrier(carriedTurn{
		Items:      sampleReasoningItems(),
		Calls:      []carriedCall{{ProxyID: "call_vekil_x", UpstreamID: "call_upstream_1", Name: "lookup", ItemIndex: 1}},
		Route:      minted,
		Projection: carriedProjectionDigest(content, projected),
	})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	replay, ok := decodeReasoningCarrier(signature, nil)
	if !ok {
		t.Fatal("decode rejected a carrier we just produced")
	}
	carried := map[string]carriedReplay{"call_vekil_x": replay}

	if _, ok := carriedRestoredCalls(carried, projected, minted, content); !ok {
		t.Fatal("the minting route did not restore its own carrier")
	}
	if _, ok := carriedRestoredCalls(carried, projected, other, content); ok {
		t.Fatal("a different route restored the carrier, so nothing binds it to its model or tier")
	}
}

// Assert on the MARSHALLED BYTES. A struct round-trip cannot see a field that
// omitempty deleted on the way out, which is how {"type":"thinking",
// "signature":...} shipped and killed clients on i.thinking.length.
func TestCarrierBlockKeepsThinkingOnTheWire(t *testing.T) {
	block, err := reasoningCarrierBlock(carriedTurn{Items: sampleReasoningItems()})
	if err != nil {
		t.Fatalf("block: %v", err)
	}
	encoded, err := json.Marshal(block)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &wire); err != nil {
		t.Fatal(err)
	}
	raw, present := wire["thinking"]
	if !present {
		t.Fatalf("no thinking key on the wire; clients dereference it:\n%s", encoded)
	}
	var thinking string
	if json.Unmarshal(raw, &thinking) != nil || thinking != "" {
		t.Fatalf("thinking = %s, want an empty string", raw)
	}
}

// Non-carrier blocks must NOT gain the field: Anthropic does not send
// `thinking` on a text block, which is why this is a pointer rather than
// dropping omitempty.
func TestNonCarrierBlocksOmitThinkingOnTheWire(t *testing.T) {
	text := "hello"
	for _, block := range []models.ContentBlock{
		{Type: "text", Text: &text},
		{Type: "tool_use", ID: "toolu_1", Name: "lookup", Input: json.RawMessage(`{}`)},
	} {
		encoded, err := json.Marshal(block)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(encoded), `"thinking"`) {
			t.Fatalf("%s block gained a thinking field: %s", block.Type, encoded)
		}
	}
}
