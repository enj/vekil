package proxy

import (
	"bytes"
	"compress/flate"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"io"
	"strings"
	"sync"

	"github.com/sozercan/vekil/models"
)

const reasoningCarrierPrefix = "vekil1."

// A signature is client input, so a compression bomb is reachable, and one body holds many.
const (
	reasoningCarrierMaxDecodedBytes = 1 << 20
	reasoningCarrierRequestBudget   = 8 << 20
	carriedDigestBytes              = 16
)

// Carried, not inferred from item order: positional binding misattaches the
// results of same-name parallel calls.
type carriedCall struct {
	ProxyID    string `json:"proxy_id"`
	UpstreamID string `json:"upstream_id"`
	Name       string `json:"name"`
	ItemIndex  int    `json:"item_index"`
}

type reasoningCarrierPayload struct {
	Items []json.RawMessage `json:"items"`
	Calls []carriedCall     `json:"calls,omitempty"`
	// Binds a carrier to the route that minted it: Copilot's ciphertext is model-bound,
	// and on the policy path this re-picks the tier. RouteTag keys that second role;
	// the digest stays unkeyed so a restart can still restore items.
	RouteDigest      string `json:"route_digest,omitempty"`
	RouteTag         string `json:"route_tag,omitempty"`
	ProjectionDigest string `json:"projection_digest,omitempty"`
}

// A turn's Responses output and bindings, encoded into an Anthropic thinking block's
// signature so the CLIENT holds it rather than the replay store, whose TTL, eviction
// and restart each wedge a conversation. Chat has no such field, so it keeps the store.
type carriedTurn struct {
	Items      []json.RawMessage
	Calls      []carriedCall
	Route      responsesChatReplayRoute
	Projection string
}

type carriedReplay struct {
	Items            []json.RawMessage
	Calls            map[string]carriedCall
	RouteDigest      string
	RouteTagValid    bool
	ProjectionDigest string
}

func carriedRouteDigest(route responsesChatReplayRoute) string {
	sum := sha256.Sum256([]byte(strings.Join([]string{
		route.ProviderID, route.PublicModel, route.UpstreamModel, route.RouteID, route.PolicyTier,
	}, "\x00")))
	return hex.EncodeToString(sum[:8])
}

// A carrier is client input, so its route claim is authority only if this process
// minted the tag. Losing the key on restart fails closed, which is the old 400.
var reasoningCarrierKey = sync.OnceValue(func() []byte {
	key := make([]byte, sha256.Size)
	if _, err := rand.Read(key); err != nil {
		return nil
	}
	return key
})

func reasoningCarrierRouteTag(digest string) string {
	key := reasoningCarrierKey()
	if key == nil || digest == "" {
		return ""
	}
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(digest))
	return hex.EncodeToString(mac.Sum(nil)[:16])
}

func reasoningCarrierRouteTagValid(digest, tag string) bool {
	expected := reasoningCarrierRouteTag(digest)
	return expected != "" && hmac.Equal([]byte(expected), []byte(tag))
}

// A tier is authority the store used to hold, so only a carrier this process tagged
// may pick one; an untagged carrier still restores items under a route already decided.
func routeSelectingCarriers(carried map[string]carriedReplay) map[string]carriedReplay {
	selecting := make(map[string]carriedReplay, len(carried))
	for id, replay := range carried {
		if replay.RouteTagValid {
			selecting[id] = replay
		}
	}
	if len(selecting) == 0 {
		return nil
	}
	return selecting
}

// Mirrors responsesChatReplayGroup.matchesProjection: same canonical assistant text,
// same calls, same order. Arguments stay out -- the store accepts several normalised
// forms of them, so binding them here would wedge a transcript the store allows.
func carriedProjectionDigest(content []byte, calls []responsesChatReplayProjectedCall) string {
	sum := sha256.New()
	sum.Write(content)
	for _, call := range calls {
		sum.Write([]byte{0})
		sum.Write([]byte(call.ID))
		sum.Write([]byte{0})
		sum.Write([]byte(call.Name))
	}
	return hex.EncodeToString(sum.Sum(nil)[:carriedDigestBytes])
}

// A digest is client input that keys a request-scoped map, so drop anything not ours.
func carriedDigest(value string) string {
	if len(value) != hex.EncodedLen(carriedDigestBytes) {
		return ""
	}
	if _, err := hex.DecodeString(value); err != nil {
		return ""
	}
	return value
}

func carriedTurnFromPublished(route responsesChatReplayRoute, outputItems []json.RawMessage, published responsesChatReplayPublished) carriedTurn {
	calls := make([]carriedCall, len(published.Calls))
	for i, call := range published.Calls {
		calls[i] = carriedCall{
			ProxyID:    call.ProxyCallID,
			UpstreamID: call.UpstreamCallID,
			Name:       call.Name,
			ItemIndex:  call.OutputItemIndex,
		}
	}
	return carriedTurn{
		Items:      outputItems,
		Calls:      calls,
		Route:      route,
		Projection: carriedProjectionDigest(published.Projection.Content, published.Projection.Calls),
	}
}

func encodeReasoningCarrier(turn carriedTurn) (string, error) {
	if len(turn.Items) == 0 {
		return "", nil
	}
	digest := carriedRouteDigest(turn.Route)
	payload, err := json.Marshal(reasoningCarrierPayload{
		Items:            turn.Items,
		Calls:            turn.Calls,
		RouteDigest:      digest,
		RouteTag:         reasoningCarrierRouteTag(digest),
		ProjectionDigest: turn.Projection,
	})
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

// Unusable carriers return false rather than erroring: failing would turn lost
// continuity into a dead conversation.
func decodeReasoningCarrier(signature string, budget *int) (carriedReplay, bool) {
	if !strings.HasPrefix(signature, reasoningCarrierPrefix) {
		return carriedReplay{}, false
	}
	compressed, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(signature, reasoningCarrierPrefix))
	if err != nil || len(compressed) == 0 {
		return carriedReplay{}, false
	}
	limit := reasoningCarrierMaxDecodedBytes
	if budget != nil {
		if *budget <= 0 {
			return carriedReplay{}, false
		}
		limit = min(limit, *budget)
	}
	reader := flate.NewReader(bytes.NewReader(compressed))
	defer func() { _ = reader.Close() }()
	payload, err := io.ReadAll(io.LimitReader(reader, int64(limit)+1))
	if budget != nil {
		*budget -= len(payload)
	}
	if err != nil || len(payload) > limit {
		return carriedReplay{}, false
	}
	var decoded reasoningCarrierPayload
	if json.Unmarshal(payload, &decoded) != nil || len(decoded.Items) == 0 {
		return carriedReplay{}, false
	}
	replay := carriedReplay{
		Items:            decoded.Items,
		Calls:            make(map[string]carriedCall, len(decoded.Calls)),
		RouteDigest:      decoded.RouteDigest,
		RouteTagValid:    reasoningCarrierRouteTagValid(decoded.RouteDigest, decoded.RouteTag),
		ProjectionDigest: carriedDigest(decoded.ProjectionDigest),
	}
	for _, call := range decoded.Calls {
		replay.Calls[call.ProxyID] = call
	}
	return replay, true
}

// `thinking` is empty on purpose: the block is transport, not content.
func reasoningCarrierBlock(turn carriedTurn) (*models.ContentBlock, error) {
	signature, err := encodeReasoningCarrier(turn)
	if err != nil || signature == "" {
		return nil, err
	}
	return &models.ContentBlock{Type: "thinking", Thinking: stringPtr(""), Signature: signature}, nil
}

// Keyed by tool_use id, not message index, because clients trim history. Every
// tool_use in a message maps to that message's carrier: one output array per turn.
// The budget is charged oldest-first, so a long transcript starves its newest turns.
func extractCarriedReasoning(messages []models.AnthropicMessage) (carried map[string]carriedReplay, starved bool) {
	carried = make(map[string]carriedReplay)
	budget := reasoningCarrierRequestBudget
	for _, msg := range messages {
		if msg.Role != "assistant" {
			continue
		}
		var blocks []models.ContentBlock
		if json.Unmarshal(msg.Content, &blocks) != nil {
			continue
		}
		var replay carriedReplay
		var toolUseIDs []string
		decodeSpent := false
		for _, block := range blocks {
			switch block.Type {
			case "thinking", "redacted_thinking":
				// Charged on the attempt, so corrupt blocks cannot each buy an inflate.
				if decodeSpent || !strings.HasPrefix(block.Signature, reasoningCarrierPrefix) {
					continue
				}
				decodeSpent = true
				if budget <= 0 {
					starved = true
					continue
				}
				if decoded, ok := decodeReasoningCarrier(block.Signature, &budget); ok {
					replay = decoded
				}
			case "tool_use":
				if block.ID != "" {
					toolUseIDs = append(toolUseIDs, block.ID)
				}
			}
		}
		if replay.Items == nil || len(toolUseIDs) == 0 {
			continue
		}
		for _, id := range toolUseIDs {
			carried[id] = replay
		}
	}
	if len(carried) == 0 {
		return nil, starved
	}
	return carried, starved
}

// A mixed group hands Copilot a chain that does not match the calls beside it.
func carriedReplayForCalls(carried map[string]carriedReplay, projected []responsesChatReplayProjectedCall) (carriedReplay, bool) {
	if len(carried) == 0 || len(projected) == 0 {
		return carriedReplay{}, false
	}
	var replay carriedReplay
	for i, call := range projected {
		found, ok := carried[call.ID]
		if !ok || len(found.Items) == 0 {
			return carriedReplay{}, false
		}
		if i == 0 {
			replay = found
			continue
		}
		if len(replay.Items) != len(found.Items) {
			return carriedReplay{}, false
		}
		for j := range replay.Items {
			if string(replay.Items[j]) != string(found.Items[j]) {
				return carriedReplay{}, false
			}
		}
	}
	return replay, true
}

// A shape we did not mint, or that the transcript cannot rebuild, restores nothing.
func carriedItemsWellShaped(items []json.RawMessage) bool {
	messages := 0
	for _, item := range items {
		var header struct {
			Type   string `json:"type"`
			Role   string `json:"role"`
			CallID string `json:"call_id"`
			Name   string `json:"name"`
		}
		if json.Unmarshal(item, &header) != nil {
			return false
		}
		switch strings.TrimSpace(header.Type) {
		case "reasoning":
		case "message":
			messages++
			if messages > 1 || strings.TrimSpace(header.Role) != "assistant" {
				return false
			}
		case "function_call":
			if strings.TrimSpace(header.CallID) == "" || strings.TrimSpace(header.Name) == "" {
				return false
			}
		default:
			return false
		}
	}
	return true
}

// The store's binding without the store: route, projection and each call's own minted id
// must match, and indices mirror Publish. "" means restored, else the guard that refused.
func carriedRestoredCalls(carried map[string]carriedReplay, projected []responsesChatReplayProjectedCall, route responsesChatReplayRoute, projectionContent json.RawMessage) (responsesChatRestoredCalls, string) {
	canonicalContent, err := canonicalReplayJSONValue(projectionContent)
	if err != nil {
		return responsesChatRestoredCalls{}, "projection"
	}
	replay, ok := carriedReplayForCalls(carried, projected)
	switch {
	case !ok:
		return responsesChatRestoredCalls{}, "absent"
	case replay.RouteDigest != carriedRouteDigest(route):
		return responsesChatRestoredCalls{}, "route"
	case replay.ProjectionDigest != carriedProjectionDigest(canonicalContent, projected):
		return responsesChatRestoredCalls{}, "projection"
	case !carriedItemsWellShaped(replay.Items):
		return responsesChatRestoredCalls{}, "shape"
	}
	calls := make([]responsesChatReplayResolvedCall, len(projected))
	lastItemIndex := -1
	for i, projectedCall := range projected {
		call, known := replay.Calls[projectedCall.ID]
		upstreamID := strings.TrimSpace(call.UpstreamID)
		if !known || upstreamID == "" || call.Name != strings.TrimSpace(projectedCall.Name) ||
			call.ItemIndex <= lastItemIndex || call.ItemIndex >= len(replay.Items) {
			return responsesChatRestoredCalls{}, "binding"
		}
		itemType, itemCallID := carriedItemHeader(replay.Items[call.ItemIndex])
		if itemType != "function_call" || itemCallID != upstreamID {
			return responsesChatRestoredCalls{}, "binding"
		}
		lastItemIndex = call.ItemIndex
		calls[i] = responsesChatReplayResolvedCall{
			ProxyCallID:     projectedCall.ID,
			UpstreamCallID:  upstreamID,
			Name:            call.Name,
			OutputItemIndex: call.ItemIndex,
			OutputItem:      replay.Items[call.ItemIndex],
		}
	}
	digest := sha256.New()
	for _, item := range replay.Items {
		digest.Write(item)
	}
	return responsesChatRestoredCalls{
		Key:         "carrier:" + replay.ProjectionDigest + ":" + hex.EncodeToString(digest.Sum(nil)),
		OutputItems: replay.Items,
		Calls:       calls,
		Carried:     true,
	}, ""
}

func carriedItemHeader(item json.RawMessage) (itemType, callID string) {
	var header struct {
		Type   string `json:"type"`
		CallID string `json:"call_id"`
	}
	if json.Unmarshal(item, &header) != nil {
		return "", ""
	}
	return strings.TrimSpace(header.Type), strings.TrimSpace(header.CallID)
}

// A reasoning item keeps only its ciphertext and id -- Copilot's own, and the one thing
// a transcript cannot rebuild. Text and calls come from the transcript the classifier read.
func reconstructCarriedRestore(restored responsesChatRestoredCalls, projected []responsesChatReplayProjectedCall, assistantText string) responsesChatRestoredCalls {
	visible := make(map[string]responsesChatReplayProjectedCall, len(projected))
	for _, call := range projected {
		visible[call.ID] = call
	}
	callByItemIndex := make(map[int]int, len(restored.Calls))
	for i, call := range restored.Calls {
		callByItemIndex[call.OutputItemIndex] = i
	}
	items := make([]json.RawMessage, 0, len(restored.OutputItems))
	calls := make([]responsesChatReplayResolvedCall, len(restored.Calls))
	copy(calls, restored.Calls)
	textSlot := carriedTextSlot(restored.OutputItems, callByItemIndex)
	for index, item := range restored.OutputItems {
		if index == textSlot && assistantText != "" {
			items = appendAssistantHistoryMessage(items, assistantText)
		}
		if i, ok := callByItemIndex[index]; ok {
			call := visible[calls[i].ProxyCallID]
			calls[i].OutputItem = responsesFunctionCallItem(calls[i].UpstreamCallID, call.Name, call.Arguments)
			calls[i].OutputItemIndex = len(items)
			items = append(items, calls[i].OutputItem)
			continue
		}
		if itemType, _ := carriedItemHeader(item); itemType == "reasoning" {
			if reasoning := carriedReasoningCiphertext(item); reasoning != nil {
				items = append(items, reasoning)
			}
		}
	}
	restored.OutputItems = items
	restored.Calls = calls
	return restored
}

// One flattened text replays where the turn first spoke: its message slot, or the first call.
func carriedTextSlot(items []json.RawMessage, callByItemIndex map[int]int) int {
	for index, item := range items {
		itemType, _ := carriedItemHeader(item)
		if _, isCall := callByItemIndex[index]; isCall || itemType == "message" {
			return index
		}
	}
	return -1
}

// summary, content and every other field are plain text no policy check ever read.
func carriedReasoningCiphertext(item json.RawMessage) json.RawMessage {
	var fields map[string]json.RawMessage
	if json.Unmarshal(item, &fields) != nil || rawJSONString(fields["encrypted_content"]) == "" {
		return nil
	}
	narrowed := map[string]json.RawMessage{
		"type":              json.RawMessage(`"reasoning"`),
		"encrypted_content": fields["encrypted_content"],
		"summary":           json.RawMessage(`[]`),
		"content":           json.RawMessage(`[]`),
	}
	if id, ok := fields["id"]; ok {
		narrowed["id"] = id
	}
	rebuilt, err := json.Marshal(narrowed)
	if err != nil {
		return nil
	}
	return rebuilt
}

func prependCarriedReasoning(resp *models.AnthropicResponse, turn carriedTurn) *models.AnthropicResponse {
	if resp == nil || len(turn.Items) == 0 {
		return resp
	}
	carrier, err := reasoningCarrierBlock(turn)
	if err != nil || carrier == nil {
		return resp
	}
	resp.Content = append([]models.ContentBlock{*carrier}, resp.Content...)
	return resp
}
