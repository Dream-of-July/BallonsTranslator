# Engine protocol (v1)

The SwiftUI app drives the Python engine (`python -m ballontranslator.engine`) as a
long-lived child process. This file is the contract between the two; change it in the
same commit as either side.

## Transport

- The app spawns the engine with the repo root as working directory and
  `BALLOONTRANS_APP_ROOT` set to it.
- **Requests**: one JSON object per line on the engine's **stdin** (UTF-8, `\n` terminated).
- **Responses and events**: one JSON object per line on the engine's **stdout**.
  The engine must guarantee nothing else is ever written to that stream (library
  `print()` output, tqdm bars and logs go to **stderr**).
- All paths are absolute. All geometry is in **source-image pixels**, origin top-left.
- The engine exits when stdin reaches EOF or on `shutdown`.

```
→ {"id": 7, "method": "open_project", "params": {"path": "/…/chapter1"}}
← {"id": 7, "result": {…}}
← {"id": 7, "error": {"code": "not_found", "message": "Folder does not exist."}}
← {"event": "stage_progress", "stage": "ocr", "done": 3, "total": 12}
```

- `id` is chosen by the app and echoed back. Exactly one response per request.
- `error.code` is a stable machine string; `error.message` is a short English sentence
  safe to show to the user. Never contains API keys.
- Events have no `id` and may arrive at any time, including between a request and its
  response.

## Methods

### `hello` → engine and environment status
Result: `{"protocol": 1, "engine_version": "1.5.21", "device": "mps"|"cpu"|"cuda"|null,
"torch_installed": bool, "models_ready": bool,
"capabilities": ["keep_original", "vision_probe", "enhance"]}`.
`capabilities` lists supported optional protocol features. `vision_probe` advertises
the separate image-capability check and the measured `Profile.vision` field.
`enhance` advertises the `enhance` settings field and the `enhance` run stage.
`models_ready` is false when the selected detector/OCR/inpainter still need a download
or package install; the first `run` will then emit `prepare_progress` events.
Sent by the app first; the engine may take several seconds to answer (imports).

### `open_project` `{path}` → project
`path` is a folder of images. Creates the project file if missing (existing behaviour).
Result: `{"path", "name", "pages": [Page…]}`.

`Page`: `{"name": "001.png", "source": "/abs/001.png", "width", "height",
"result": "/abs/result/001.png"|null, "inpainted": "/abs/…"|null,
"done": bool, "revision": int, "blocks": [Block…]}`.
`revision` increases every time the page's result image is rewritten (use it to bust
image caches).

`Block`: `{"index": int, "rect": [x, y, w, h], "angle": deg,
"source_text": str, "translation": str, "vertical": bool,
"font_family": str, "font_size": px, "color": [r,g,b], "alignment": "left"|"center"|"right",
"bold": bool, "italic": bool}`.
`rect` is the unrotated render box. `font_size` is in image pixels.
`source_rect` is the optional original detection box in the same coordinates.
`review_reason` is optional and can be `unreadable`, `untranslated`,
`geometry`, `overflow`, `protected` or `skipped`. Withheld regions keep their source pixels.
`skipped` marks text outside any speech bubble that could not be read (artwork, sound
effects, signatures): it is left as the original and is not an item to review. The app
lets the user type its text and sends it as `source_text` (with `autofit`) to have it translated.

### `get_page` `{name}` → Page
Fresh copy of one page.

### `run` `{pages?: [name…], mode: "all"|"continue"}` → `{"started": true}`
Starts the full pipeline (detect → OCR → translate → inpaint → render) for the given
pages (default: all). `continue` skips pages already done. Returns immediately; progress
arrives as events and completion as `run_finished`. Only one run at a time
(`error.code = "busy"`). Fails fast with `error.code = "translator_not_ready"` and
`error.data = {"missing": "api_key"|"model"|"base_url"}` when the selected translator
cannot work.

### `cancel` → `{"cancelling": true}`
Requests a stop. Takes effect between stages/pages; `run_finished` follows with
`"cancelled": true`.

### `update_blocks` `{page, edits: [{index, source_text?, translation?, rect?, angle?, font_family?,
font_size?, color?, alignment?, bold?, italic?, keep_original?}], autofit?: bool}` → Page
Applies the edits, re-renders **that page only**, saves the project and the result
image, and returns the updated Page (new `revision`). `autofit: true` re-runs the
engine's automatic layout for the edited blocks (may change their `rect` and
`font_size`; the returned Page carries the final values). Rejected while a run is in
progress (`busy`).
`source_text` records a manual OCR correction. A Japanese page analysed by the
native engine keeps its independently evaluated cleanup mask through edits.
`keep_original` (implemented; optional bool in an edit):
`true` leaves a withheld block untranslated on purpose, so its `review_reason` becomes
`protected` and the source pixels stay; `false` withdraws that and the block is evaluated
again. The engine lists `"keep_original"` in the `capabilities` string array of its
`hello` result; the app offers the action only then.

### `delete_blocks` `{page, indexes: [int…]}` → Page
Removes blocks (their text is no longer drawn) and re-renders.

### `list_fonts` → `{"fonts": [{"family": str, "display": str, "source": "system"|"custom"}], "default": str}`
`family` is the value to send back in `font_family`.

### `get_settings` → Settings
```
{"source_language": "日本語", "target_language": "简体中文",
 "languages": {"source": [str…], "target": [str…]},
 "translator": {"kind": "google"|"llm"|"other", "module": str, "profile_id": str|null},
 "profiles": [Profile…],
 "provider_templates": [str…],
 "output": {"format": ".png"|".jpg"|".webp", "quality": int},
 "enhance": {"enabled": bool, "profile_id": str|null}}
```
`enhance` (capability `enhance`; absent on older engines, read as off): Enhanced mode.
After a page is composed, hard text blocks are re-checked by a vision model — image
crops of the page are sent to the chosen service — and corrected once. Off by default.
`profile_id: null` uses the translation service's profile. The app only offers to turn
it on when that effective profile's `vision` is `yes`. The engine also enforces
this measured capability: no automatic probes, and no images when disabled or
when the effective profile is missing/unmeasured/negative. A null profile has no
effect for a non-LLM translator. One `log` event per run explains a skipped mode.

On the Japanese manga quality path, only suspicious text blocks are sent to the
chosen profile: withheld unreadable/geometry/overflow/untranslated blocks, at
most six largest skipped blocks, and drawn blocks with uncertain OCR, complex
cleanup or legibility adjustments. There are at most 24 blocks per page, two
margin crops per block (original and composed, each at most 512 pixels on its
long side), and one request, or two batches when more than 12 blocks qualify.
Clean pages send nothing. Prompts contain block indexes, OCR and translation;
the engine adds no file paths or project names. Translation and OCR remain
text/local-image operations; only this opt-in review sends page crops.

There is one review round. Its total per-page budget, including correction and
local reprocessing, is the LLM request timeout (120 seconds by default). Advice
is type-checked and limited to requested indexes. Corrected source text is
translated through the existing text-only path; local fit, collision and
residue checks remain authoritative. Non-dialogue keeps original pixels and
reports `skipped`. Rejected changes preserve the pre-review result; a locally
confirmed erase failure restores original pixels and withholds the affected
block with its normal review reason. A timeout,
HTTP error, invalid JSON or missing required response field discards the page's
review, logs a sanitized reason, and preserves the ordinary result without an
`error` event or failed page. Cancel interrupts review without publishing late
results. Logs give checked/corrected/re-erased/restyled/non-dialogue/abandoned
counts. `stage_progress(enhance)` precedes the page's single `page_done`, which
still means the final image was written. Block project records retain a
default-false `manga_enhanced` flag when an accepted review changed that block.
`Profile`: `{"id", "name", "base_url", "model", "model_options": [str…],
"require_api_key": bool, "has_api_key": bool, "vision": "yes"|"no"|"unknown"}` —
the key itself is never sent. `vision` is the persisted measurement from
`probe_vision`, never a provider-template guess. New and older unmeasured profiles
report `unknown`; the app reads a missing `vision` as `unknown`.

### `set_settings` `{source_language?, target_language?, translator?: {"kind": "google"}
| {"kind": "llm", "profile_id"}, output?: {...}, enhance?: {enabled?, profile_id?}}` → Settings
In `enhance`, omitted keys are left untouched and `profile_id: null` selects the
translation service's profile; an unknown enhance `profile_id` is `bad_request`.

### `create_profile` `{template}` → Profile
New LLM profile from a provider template (see `provider_templates`).

### `update_profile` `{id, name?, base_url?, model?, api_key?}` → Profile
`api_key` is write-only; pass `""` to clear it.
Changing `base_url` or `model` resets the vision measurement to `unknown`.

### `delete_profile` `{id}` → Settings

### `test_profile` `{id}` → `{"ok": bool, "message": str, "kind": str, "models": [str…]}`
Network call on a worker thread; other requests keep being served meanwhile.
This connection check does not send an image.

### `probe_vision` `{profile_id}` → `{"vision": "yes"|"no"}`
Capability `vision_probe`.
Runs on a worker thread using the existing connection-test request path and its
15-second timeout. Sends exactly one chat request with a small synthetic image
generated in memory; no user image or disk file is used. Uses `vision_model` if
configured, otherwise `model`, and disables thinking when the profile allows it.
Only a reply containing the exact random four digits drawn in the image yields
`yes`; a wrong/empty reply or image request rejection (HTTP 400/415/422) yields `no`.
The result survives restart and synchronizes the profile's `support_vision` flag.

Authentication, network, timeout and other connection failures return the same
failure result shape as `test_profile`:
`{"ok": false, "message": "Connection failed.", "kind": str, "models": []}`.
They leave the stored measurement unchanged. Raw provider errors are never returned.
Invalid parameters and missing profiles use the usual protocol `error` envelope.
If the profile's endpoint/model changes during a probe, its result is discarded
with `error.code = "busy"`; a deleted profile returns `not_found`.
Translation and OCR keep their existing local-image behavior; this method alone
explicitly sends the synthetic probe image.

### `export` `{destination, what: "images"|"translation_txt"|"source_txt"}` → `{"files": [path…]}`
`images` copies every existing result image into `destination` (created if needed),
keeping file names. Text exports write one file into `destination`.

### `shutdown` → `{"ok": true}` then the process exits.

## Events

| event | fields |
|---|---|
| `prepare_progress` | `module` (`detector`/`ocr`/`inpainter`/`translator`/`packages`), `phase` (e.g. `downloading`, `installing`, `loading`), `message` (short English), `done`?, `total`? (bytes) |
| `stage_progress` | `stage` (`detect`/`ocr`/`translate`/`inpaint`/`enhance`), `done`, `total` (pages). `enhance` arrives only while enhanced mode re-checks pages |
| `page_done` | `page` (name), `revision` — result image written; fetch with `get_page` |
| `run_finished` | `cancelled`: bool, `failed_pages`: [name…] |
| `error` | `message`, `page`? — non-fatal problem during a run (e.g. translation failed for a page) |
| `log` | `level`, `message` — optional, for a diagnostics window |

## Error codes

`bad_request`, `not_found`, `busy`, `no_project`, `translator_not_ready`, `internal`.

## Engine notes

- `BALLOONTRANS_ENHANCE_DEBUG_DIR`: opt-in existing directory for one text-only JSON diagnostic per page (request text, raw model reply, parsed verdicts and block actions; no images, credentials or headers); unset by default.
- On macOS the engine pins Qt's offscreen font DPI to 72 and sets `shared.LDPI`
  from that screen. Both sides of the px/pt conversion must agree; changing only
  `LDPI` changes glyph size. This matches the normal Cocoa 72-DPI point convention
  and preserves GUI HTML point sizes. Projects authored with a custom GUI `--ldpi`
  or on another platform carry no originating DPI metadata; identical rich-text
  sizing across those environments cannot be inferred automatically.
- `models_ready` checks declared dependencies and local model-file presence without
  loading models. File integrity is verified by the existing preparation path on
  `run`. Raw provider exception text is omitted from protocol errors and logs.
- `revision` counts successful engine image writes within the current open project;
  opening a project starts a new revision sequence. Cancellation waits for active
  inference or package installation to leave its worker before `run_finished`.
