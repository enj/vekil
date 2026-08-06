package proxy

// Carrier for Responses reasoning items, encoded into an Anthropic
// `thinking` block's `signature` so the CLIENT holds them across turns.
//
// WHY
//
// Copilot exposes gpt-5.x only via the Responses API, which requires the
// opaque reasoning items it returns to be handed back on the next turn.
// Those items have no representation in the Chat or Anthropic wire formats,
// so the proxy used to mint `call_vekil_*` tool-call ids and park the real
// payload in an in-memory LRU with a one-hour TTL.
//
// That store forgot on idle gaps, on eviction, and on every restart — and a
// forgotten group was unrecoverable, because the ids live in the client's
// transcript forever and re-resolve on every later request. Measured: Copilot
// itself accepts the same reasoning items at least 131 minutes later, so the
// expiry was the proxy's own bookkeeping, not an upstream constraint.
//
// The client already persists opaque provider state: Anthropic `thinking`
// blocks carry a `signature` that clients store and replay verbatim. Putting
// the items there removes the need for any server-side state.
//
// NO ENCRYPTION HERE, DELIBERATELY
//
// The payload is already ciphertext. A reasoning item is
// `{content: [], summary: [], encrypted_content: <ciphertext>, id: <opaque>}`
// — there is no plaintext to protect, and sealing it again would only add a
// key to manage. Integrity is upstream's: tamper with `encrypted_content` and
// Copilot's own decryption fails. Compression is for size, not secrecy.

import (
	"bytes"
	"compress/flate"
	"encoding/base64"
	"encoding/json"
	"io"
	"strings"

	"github.com/sozercan/vekil/models"
)

// reasoningCarrierPrefix version-tags the signature. An unrecognised prefix
// must be IGNORED rather than rejected: a client may be replaying a signature
// minted by a different proxy, a future version, or genuine Anthropic
// extended-thinking state. None of those are errors, and 400ing on them would
// reintroduce the wedge this replaces.
const reasoningCarrierPrefix = "vekil1."

// reasoningCarrierMaxDecodedBytes bounds what a decode will inflate. The
// signature arrives from a client, so a compression bomb is reachable; 8 MiB
// is far above any real turn (~2 KB per reasoning item) and far below memory
// pressure.
const reasoningCarrierMaxDecodedBytes = 8 << 20

// encodeReasoningCarrier packs Responses output items into a signature value.
//
// Returns "" when there is nothing worth carrying, so callers can simply omit
// the thinking block rather than emit an empty one.
func encodeReasoningCarrier(outputItems []json.RawMessage) (string, error) {
	if len(outputItems) == 0 {
		return "", nil
	}
	payload, err := json.Marshal(outputItems)
	if err != nil {
		return "", err
	}
	var compressed bytes.Buffer
	writer, err := flate.NewWriter(&compressed, flate.BestCompression)
	if err != nil {
		return "", err
	}
	if _, err := writer.Write(payload); err != nil {
		_ = writer.Close()
		return "", err
	}
	if err := writer.Close(); err != nil {
		return "", err
	}
	return reasoningCarrierPrefix + base64.RawURLEncoding.EncodeToString(compressed.Bytes()), nil
}

// decodeReasoningCarrier unpacks a signature previously produced by
// encodeReasoningCarrier.
//
// The bool reports whether this signature was ours and decoded cleanly.
// Everything else — foreign prefix, corrupt base64, bad deflate, oversized
// payload, wrong JSON shape — returns false with no error, because the only
// sane response to an unusable carrier is to proceed without it. Failing the
// request instead would turn a recoverable quality loss into exactly the dead
// conversation this design exists to prevent.
func decodeReasoningCarrier(signature string) ([]json.RawMessage, bool) {
	if !strings.HasPrefix(signature, reasoningCarrierPrefix) {
		return nil, false
	}
	encoded := strings.TrimPrefix(signature, reasoningCarrierPrefix)
	compressed, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return nil, false
	}
	reader := flate.NewReader(bytes.NewReader(compressed))
	defer func() { _ = reader.Close() }()
	payload, err := io.ReadAll(io.LimitReader(reader, reasoningCarrierMaxDecodedBytes+1))
	if err != nil {
		return nil, false
	}
	if len(payload) > reasoningCarrierMaxDecodedBytes {
		return nil, false
	}
	var items []json.RawMessage
	if err := json.Unmarshal(payload, &items); err != nil || len(items) == 0 {
		return nil, false
	}
	return items, true
}

// reasoningCarrierBlock renders the Anthropic content block that transports
// the carrier.
//
// `thinking` is deliberately empty. Clients already store thinking blocks
// whose text is empty and whose signature is opaque, so this adds nothing
// visible to the user — the block is transport, not content.
func reasoningCarrierBlock(outputItems []json.RawMessage) (map[string]any, error) {
	signature, err := encodeReasoningCarrier(outputItems)
	if err != nil || signature == "" {
		return nil, err
	}
	return map[string]any{
		"type":      "thinking",
		"thinking":  "",
		"signature": signature,
	}, nil
}

// extractCarriedReasoning collects, for every assistant tool_use id in an
// Anthropic request, the Responses output items carried alongside it.
//
// Runs as a separate pass rather than threading a return value through
// translateMessage/TranslateAnthropicToOpenAI. Those functions build the Chat
// message body, and a thinking block is transport rather than content — it
// still has no place in an OpenAI message, so they keep skipping it and their
// signatures stay put.
//
// Keyed by tool_use id because a conversation has many assistant turns, each
// with its own reasoning, and message indices shift the moment a client trims
// history. Every tool_use in one assistant message maps to that message's
// carrier: the Responses output array covers the whole turn, including
// parallel calls.
//
// Messages without a carrier are simply absent from the map. The caller
// degrades for those, which is what makes legacy transcripts — recorded
// before this existed, or by a client that drops thinking blocks — keep
// working instead of failing.
func extractCarriedReasoning(messages []models.AnthropicMessage) map[string][]json.RawMessage {
	carried := make(map[string][]json.RawMessage)
	for _, msg := range messages {
		if msg.Role != "assistant" {
			continue
		}
		var blocks []models.ContentBlock
		if err := json.Unmarshal(msg.Content, &blocks); err != nil {
			continue // string content: no blocks, nothing to carry
		}
		var items []json.RawMessage
		var toolUseIDs []string
		for _, block := range blocks {
			switch block.Type {
			case "thinking", "redacted_thinking":
				if items != nil {
					continue // first carrier in a message wins
				}
				if decoded, ok := decodeReasoningCarrier(block.Signature); ok {
					items = decoded
				}
			case "tool_use":
				if block.ID != "" {
					toolUseIDs = append(toolUseIDs, block.ID)
				}
			}
		}
		if items == nil {
			continue
		}
		for _, id := range toolUseIDs {
			carried[id] = items
		}
	}
	if len(carried) == 0 {
		return nil
	}
	return carried
}

// carriedReasoningGroupID marks a turn restored from a client-carried
// signature rather than the store. The store uses uint64 group ids to reject a
// group appearing twice in one request; carried turns need the same guard but
// have no store id, so they share this sentinel.
const carriedReasoningGroupID uint64 = 0

// carriedItemsForCalls returns the carried output items covering EVERY
// projected call in an assistant turn.
//
// All-or-nothing on purpose. A partially-carried turn would hand Copilot a
// reasoning chain that does not match the calls beside it, which is worse than
// having no chain at all: better to fall through and synthesise the whole turn
// consistently.
//
// Also requires every call to resolve to the SAME items. Calls from one
// assistant message share one output array, so a mismatch means the client
// stitched turns together and the carrier can no longer be trusted.
func carriedItemsForCalls(carried map[string][]json.RawMessage, projected []responsesChatReplayProjectedCall) ([]json.RawMessage, bool) {
	if len(carried) == 0 || len(projected) == 0 {
		return nil, false
	}
	var items []json.RawMessage
	for _, call := range projected {
		found, ok := carried[call.ID]
		if !ok || len(found) == 0 {
			return nil, false
		}
		if items == nil {
			items = found
			continue
		}
		if len(items) != len(found) {
			return nil, false
		}
		for i := range items {
			if string(items[i]) != string(found[i]) {
				return nil, false
			}
		}
	}
	return items, true
}
