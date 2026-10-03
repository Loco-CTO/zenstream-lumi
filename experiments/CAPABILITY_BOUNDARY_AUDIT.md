# Capability and playback boundary audit

**Audit date:** 2026-10-03

**Status:** Source audit only. No Lumi runtime or ZenStream calls exist, and the capability registry remains a draft.

## Revisions inspected

- Orchestrator: `Loco-CTO/zenstream-orchestrator` PR #261 head `59cebf8cca31b0ad08a399f08d81dc8b1527ded0`; `contracts/openapi.json` SHA-256 `bc179aa6857a062630d2fb1551aa887816711e56613342fdd7b737b49f0c403c`; `orchestrator/api/zenstream/client_routes.py` SHA-256 `2dcc411882c2ba2779655ab0006f7c75a9e090a65c0c77985cd4a2bdef33a5df`.
- Web client: `Loco-CTO/zenstream` `origin/main` ref `ff37bb4543be73f2203704959dd34f0f80546b0b`. The local `main` checkout was three commits behind; the relevant playback files had no diff against that local `origin/main` ref. Audited files: `lib/syncplay-playback.ts` SHA-256 `6c9e507aa3b9563d0777007acba96885734ace754a8725276a7cd8790d42586a`, `components/pages/player-page.tsx` SHA-256 `d36e1bc05d792801603803ca50db44b3ed435fc0a4a55a4a4f0368457c019b90`, and `components/audio/audio-player-provider.tsx` SHA-256 `22bb3d28e3e6b7aa1414e81b669ddef0385b188e2d81d623d8b5cac214792c2c`.
- Android client: `Loco-CTO/zenstream-mobile` commit `e47ef524f7925da2760ad9587fd6e6653ba2b99b`. Audited files: `app/src/main/java/com/zenstream/zenstreammobile/data/CatalogApi.kt` SHA-256 `2eac061c8b3b293883cdad70535716a3d254dc9c622d5a05d2c9752dbd125b49` and `app/src/main/java/com/zenstream/zenstreammobile/audio/AudioPlaybackService.kt` SHA-256 `b3b6896ab02cfd43c9fed32e72061968dc629a6e90770417894bc6a1d66465f`.

These pins identify the source snapshots inspected for this audit; they do not imply a Lumi integration or an approved capability contract. PR #261 corrects the OpenAPI play-start request to match the live audio client payload and includes a regression test for the required request ID.

## What the current service contract does

- `GET /api/catalog/search` requires `query` (1–200 characters), applies the authenticated account's library grants, and accepts paging plus an optional `type`. The API does not enumerate allowed `type` values and does not expose genre, release year, runtime, or watched-state filters on this operation.
- `GET /api/catalog/home` exposes named sections and an optional library ID. Results are account-filtered. A Home row is current server state, not evidence that an item is playing now.
- `GET /api/catalog/items/{entity_id}` and `/detail` return grant-filtered catalog data. An entity ID must be taken from trusted catalog output or a separately pinned context fixture; resolving a title from user text is a search task, not path construction.
- `POST /api/playback/items/{entity_id}/negotiate` authenticates the caller and asks the playback service to validate access and return negotiated media sources and access data for the client. Its request describes client playback capabilities. It does not instruct the web or Android player to start.
- `POST /api/catalog/items/{entity_id}/play-start` calls `catalog.record_play_start` with the authenticated account, entity ID, and event payload. The implementation accepts audio tracks only, requires a client-generated `playbackInstanceId` between 1 and 200 characters, and idempotently records a track play event for Watch History when enabled. The corrected OpenAPI contract now documents that required field. This is not a player command or a general video-playback operation.

The registry's `playback.start` entry therefore describes a multi-component user capability, not one Orchestrator endpoint. Its listed operations are relevant pieces of the flow, but none sends a start command to the active client.

## What the clients do

- In the web client, `useSyncplayPlayback.startPlayback` receives a catalog item. A Series is resolved to the first unwatched episode, falling back to the first ordered episode. When SyncPlay is active and controllable, the client sends its `media` command, then navigates to `/play/{entity_id}`. For video, the player page owns media negotiation and actual playback. Web audio is a separate queue owned by `audio-player-provider.tsx`.
- Android owns its playback flow in the client. `CatalogApi` negotiates media data; the video playback UI or `AudioPlaybackService` controls the appropriate player. Android audio reports the idempotent track play event through the same play-start route after client playback begins. The Orchestrator does not select a device-local player on Lumi's behalf.

The web Series-card rule is existing UI behavior. It should not be silently treated as the approved conversational rule for a request such as “play this series”; that behavior still needs to be decided and evaluated.

## Constraints for a future Lumi boundary

1. Lumi may propose a named capability and typed arguments; it must not emit arbitrary URLs, HTTP paths, SQL, shell text, or executable client commands.
2. Resolve user-mentioned titles with the authenticated client's grant-filtered catalog results. Bind the chosen stable entity ID to that result and the current request/session context; never turn model-generated title text directly into an entity path.
3. Refresh state before acting when the proposal depends on “current”, “next”, “that one”, or another mutable reference. Ask a concise clarification when the target remains ambiguous or stale.
4. Keep Orchestrator credentials, playback URLs, access tickets, lease tokens, and other playback capabilities out of model input and output. Let the authenticated client negotiate and hold those values.
5. Keep the registry at `draft` and `playback.start` at `proposed` until the product's confirmation, ambiguity, media-type routing, and error behavior are specified and covered by reviewed cases. Server authorization and the active player remain authoritative.

No capability is approved by this source audit. In particular, the audit does not define whether a direct “play …” request needs a second confirmation, how conversational Series selection should work, or which other player controls belong in Lumi's initial scope.

## Provenance boundary

This audit used only the public source code and contracts identified above. It admitted no user or catalog records, did not inspect any private library state, and does not change training, tokenizer, or evaluation data status.
