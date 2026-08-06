package proxy

// Durable backing store for Responses replay groups.
//
// WHY THIS EXISTS
//
// The in-memory store (responses_chat_replay.go) is an LRU with a one-hour
// TTL. Copilot exposes gpt-5.x only via the Responses API, which requires
// opaque reasoning items to be handed back on the next turn; those items do
// not exist in the Chat/Anthropic wire formats, so the proxy mints
// `call_vekil_*` tool-call ids and parks the real payload in that store.
//
// The consequence is that a conversation dies when the store forgets:
//
//   - idle longer than the TTL,
//   - LRU eviction under load,
//   - ANY proxy restart, which drops every in-flight conversation.
//
// None of those are the client's doing, and none are recoverable: the ids
// live in the client's transcript forever, so every later request re-resolves
// them and re-fails. Measured upstream behaviour says this is self-inflicted —
// a captured `reasoning.encrypted_content` blob replayed verbatim to Copilot
// was accepted at 25/50/70/95/131 minutes, well past both the TTL and the gap
// that killed a real session. Only the proxy's own bookkeeping expired.
//
// So: keep the LRU as a cache, and back it with files that outlive both the
// TTL and the process. On an in-memory miss we rehydrate from disk.
//
// GROWTH. Records are never pruned. A group is a couple of KB of ciphertext
// and a handful of opaque ids; even heavy daily use adds single-digit MB a
// year, which is not worth the risk of deleting state a live conversation
// still needs. Delete the directory by hand if it ever matters.
//
// PRIVACY. What lands on disk is what Copilot returned: `encrypted_content`
// is ciphertext the proxy cannot read, and the sibling fields are opaque ids.
// This persists bytes, not readable chain-of-thought.
//
// SCOPE. This is a deliberately small, local change and is NOT intended for
// upstream. The proper fix is to stop discarding thinking blocks in
// translator.go and carry reasoning items in the transcript, so no
// server-side state exists at all.

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// persistedReplayCall mirrors responsesChatReplayStoredCall. That type uses
// unexported fields (correctly — it is internal), so serialisation needs an
// explicit mirror rather than reflection over the original.
type persistedReplayCall struct {
	ProxyCallID     string `json:"proxy_call_id"`
	UpstreamCallID  string `json:"upstream_call_id"`
	Name            string `json:"name"`
	VisibleHash     string `json:"visible_hash"`
	OriginalHash    string `json:"original_hash"`
	OutputItemIndex int    `json:"output_item_index"`
}

type persistedReplayGroup struct {
	Version          int                   `json:"version"`
	ProviderID       string                `json:"provider_id"`
	PublicModel      string                `json:"public_model"`
	UpstreamModel    string                `json:"upstream_model"`
	AssistantContent []byte                `json:"assistant_content"`
	OutputItems      []json.RawMessage     `json:"output_items"`
	Calls            []persistedReplayCall `json:"calls"`
	CreatedAtUnix    int64                 `json:"created_at_unix"`
	ByteSize         int                   `json:"byte_size"`
}

const persistedReplayGroupVersion = 1

// replayPersistCallFilename maps a proxy call id to its on-disk name. The id is
// proxy-generated and charset-constrained, but it reaches this function from a
// client request, so hash it rather than trusting it in a path — a crafted id
// must never escape the directory.
func replayPersistCallFilename(callID string) string {
	sum := sha256.Sum256([]byte(callID))
	return "c-" + hex.EncodeToString(sum[:16]) + ".json"
}

// persistGroup writes a group so it can outlive eviction, expiry and restart.
// Errors are returned for the caller to log, never to fail the request on:
// a request that cannot be persisted should still be answered.
func (s *responsesChatReplayStore) persistGroup(group *responsesChatReplayGroup) error {
	if s == nil || s.persistDir == "" || group == nil {
		return nil
	}
	record := persistedReplayGroup{
		Version:          persistedReplayGroupVersion,
		ProviderID:       group.route.ProviderID,
		PublicModel:      group.route.PublicModel,
		UpstreamModel:    group.route.UpstreamModel,
		AssistantContent: group.assistantContent,
		OutputItems:      group.outputItems,
		CreatedAtUnix:    group.createdAt.Unix(),
		ByteSize:         group.byteSize,
	}
	for _, call := range group.calls {
		record.Calls = append(record.Calls, persistedReplayCall{
			ProxyCallID:     call.proxyCallID,
			UpstreamCallID:  call.upstreamCallID,
			Name:            call.name,
			VisibleHash:     hex.EncodeToString(call.visibleHash[:]),
			OriginalHash:    hex.EncodeToString(call.originalHash[:]),
			OutputItemIndex: call.outputItemIndex,
		})
	}
	encoded, err := json.Marshal(record)
	if err != nil {
		return fmt.Errorf("marshal replay group: %w", err)
	}
	if err := os.MkdirAll(s.persistDir, 0o700); err != nil {
		return fmt.Errorf("create replay dir: %w", err)
	}
	// One file per call id, each holding the whole group. Groups are capped at
	// maxCalls and the duplication buys a single stat+read on the lookup path
	// with no index file to keep consistent — worth it for a cache whose reads
	// happen on every tool-call turn.
	for _, call := range group.calls {
		path := filepath.Join(s.persistDir, replayPersistCallFilename(call.proxyCallID))
		tmp, err := os.CreateTemp(s.persistDir, ".tmp-replay-*")
		if err != nil {
			return fmt.Errorf("create temp replay file: %w", err)
		}
		tmpName := tmp.Name()
		if _, err := tmp.Write(encoded); err != nil {
			_ = tmp.Close()
			_ = os.Remove(tmpName)
			return fmt.Errorf("write replay file: %w", err)
		}
		if err := tmp.Close(); err != nil {
			_ = os.Remove(tmpName)
			return fmt.Errorf("close replay file: %w", err)
		}
		if err := os.Chmod(tmpName, 0o600); err != nil {
			_ = os.Remove(tmpName)
			return fmt.Errorf("chmod replay file: %w", err)
		}
		// Rename is atomic within a directory, so a concurrent reader sees
		// either the old file or the complete new one, never a partial write.
		if err := os.Rename(tmpName, path); err != nil {
			_ = os.Remove(tmpName)
			return fmt.Errorf("rename replay file: %w", err)
		}
	}
	return nil
}

// loadPersistedGroup rehydrates a group from disk by proxy call id. Returns
// (nil, nil) when there is simply nothing stored — a miss is not an error.
func (s *responsesChatReplayStore) loadPersistedGroup(callID string) (*responsesChatReplayGroup, error) {
	if s == nil || s.persistDir == "" {
		return nil, nil
	}
	path := filepath.Join(s.persistDir, replayPersistCallFilename(callID))
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read replay file: %w", err)
	}
	var record persistedReplayGroup
	if err := json.Unmarshal(raw, &record); err != nil {
		return nil, fmt.Errorf("decode replay file: %w", err)
	}
	if record.Version != persistedReplayGroupVersion {
		return nil, fmt.Errorf("unsupported replay record version %d", record.Version)
	}
	group := &responsesChatReplayGroup{
		route: responsesChatReplayRoute{
			ProviderID:    record.ProviderID,
			PublicModel:   record.PublicModel,
			UpstreamModel: record.UpstreamModel,
		},
		assistantContent: record.AssistantContent,
		outputItems:      record.OutputItems,
		createdAt:        time.Unix(record.CreatedAtUnix, 0),
		byteSize:         record.ByteSize,
	}
	for _, call := range record.Calls {
		stored := responsesChatReplayStoredCall{
			proxyCallID:     call.ProxyCallID,
			upstreamCallID:  call.UpstreamCallID,
			name:            call.Name,
			outputItemIndex: call.OutputItemIndex,
		}
		visible, err := hex.DecodeString(call.VisibleHash)
		if err != nil || len(visible) != sha256.Size {
			return nil, fmt.Errorf("replay record has malformed visible hash")
		}
		original, err := hex.DecodeString(call.OriginalHash)
		if err != nil || len(original) != sha256.Size {
			return nil, fmt.Errorf("replay record has malformed original hash")
		}
		copy(stored.visibleHash[:], visible)
		copy(stored.originalHash[:], original)
		group.calls = append(group.calls, stored)
	}
	return group, nil
}

// rehydrateLocked loads a group from disk and reinserts it into the live LRU
// so subsequent lookups in the same request (and later turns) hit memory.
//
// Caller must hold s.mu.
//
// The reinserted group gets a fresh expiry rather than its original one: the
// durable copy is the source of truth for whether it still exists, and giving
// it a stale expiresAt would have expireLocked drop it again immediately,
// producing exactly the wedge this is here to prevent.
func (s *responsesChatReplayStore) rehydrateLocked(callID string) (*responsesChatReplayGroup, error) {
	group, err := s.loadPersistedGroup(callID)
	if err != nil || group == nil {
		return nil, err
	}
	// A durable record can outlive the ids the process handed out, so mint a
	// fresh in-memory group id rather than trusting one from disk.
	group.id = s.nextGroupIDLocked()
	now := s.now()
	group.createdAt = now
	group.expiresAt = now.Add(s.ttl)
	group.lruElement = s.lru.PushBack(group.id)
	s.groups[group.id] = group
	for _, call := range group.calls {
		s.callsByID[call.proxyCallID] = responsesChatReplayCallRef{groupID: group.id}
	}
	s.totalBytes += group.byteSize
	// Deliberately NOT enforceLimitsLocked() here: evicting the group we just
	// restored — or the caller's other groups mid-resolve — would defeat the
	// rehydration. The next Publish enforces limits normally.
	return group, nil
}
