package proxy

// Vestiges of the Responses replay store, which is gone.
//
// WHAT WAS HERE
//
// Copilot exposes gpt-5.x only via the Responses API, which requires the
// opaque reasoning items it returns to be handed back on the next turn. Those
// items have no representation in the Chat or Anthropic wire formats, so the
// proxy minted `call_vekil_*` tool-call ids and parked the payload in an
// in-memory LRU (1h TTL, 2048 groups, 64 MiB).
//
// A conversation died whenever that store forgot: an idle gap past the TTL,
// eviction under load, or any restart. None were recoverable, because the ids
// live in the client transcript forever and re-resolved on every later request:
//
//     400 Responses-backed tool state is no longer available;
//         restart the assistant tool-call turn.
//
// Measured: Copilot accepts the same reasoning items at least 131 minutes
// later, so the expiry was the proxy's own bookkeeping, not an upstream
// constraint. The items now ride in the client transcript inside an Anthropic
// thinking block (see reasoning_carrier.go), so no server-side state exists and
// those three failure modes are gone rather than mitigated.
//
// WHAT REMAINS, AND WHY
//
// Only small value types and the legacy-id detector. The `call_vekil_` prefix
// still has to be RECOGNISED — transcripts minted before this change carry
// those ids forever, and they must route to the Responses backend and then
// degrade into a synthesised turn rather than fail. Nothing mints them any
// more.
//
// The names still say "Replay". Renaming them is a mechanical follow-up; it
// would touch seven files and was kept out of the deletion to keep this diff
// reviewable.

import ()

const (
	// legacyProxyCallIDPrefix is no longer minted, only detected: a
	// client transcript can still hold ids from before the carrier existed.
	legacyProxyCallIDPrefix = "call_vekil_"

	legacyProxyCallIDLength = len(legacyProxyCallIDPrefix) + 22
)

const (
	legacyProxyCallIDMissingCode    = "responses_replay_state_missing"
	legacyProxyCallIDMixedCode      = "responses_replay_group_mismatch"
	legacyProxyCallIDProjectionCode = "responses_replay_projection_mismatch"
)

const (
	legacyProxyCallIDMixedMessage = "Assistant tool calls reference multiple Responses replay groups."
)

// responsesChatRoute identifies which provider/model a turn belongs to.
// Kept because routing still needs it; it never had anything to do with the
// store's lifetime.
type responsesChatRoute struct {
	ProviderID    string
	PublicModel   string
	UpstreamModel string
}

func (r responsesChatRoute) equal(other responsesChatRoute) bool {
	return r.ProviderID == other.ProviderID &&
		r.PublicModel == other.PublicModel &&
		r.UpstreamModel == other.UpstreamModel
}

// responsesChatProjectedCall is one tool call as the client presented it.
// The carrier keys on these ids (see carriedItemsForCalls).
type responsesChatProjectedCall struct {
	ID        string
	Name      string
	Arguments string
}

// (cloneRawMessages lived here as a near-duplicate of the identical helper in
// responses_websocket.go -- same semantics, same append-into-nil clone. The
// rename collided with it and exposed the duplication, so this copy is gone
// and callers use the existing one.)
