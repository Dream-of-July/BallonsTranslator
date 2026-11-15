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
"torch_installed": bool, "models_ready": bool}`.
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

### `update_blocks` `{page, edits: [{index, translation?, rect?, angle?, font_family?,
font_size?, color?, alignment?, bold?, italic?}], autofit?: bool}` → Page
Applies the edits, re-renders **that page only**, saves the project and the result
image, and returns the updated Page (new `revision`). `autofit: true` re-runs the
engine's automatic layout for the edited blocks (may change their `rect` and
`font_size`; the returned Page carries the final values). Rejected while a run is in
progress (`busy`).

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
 "output": {"format": ".png"|".jpg"|".webp", "quality": int}}
```
`Profile`: `{"id", "name", "base_url", "model", "model_options": [str…],
"require_api_key": bool, "has_api_key": bool}` — the key itself is never sent.

### `set_settings` `{source_language?, target_language?, translator?: {"kind": "google"}
| {"kind": "llm", "profile_id"}, output?: {...}}` → Settings

### `create_profile` `{template}` → Profile
New LLM profile from a provider template (see `provider_templates`).

### `update_profile` `{id, name?, base_url?, model?, api_key?}` → Profile
`api_key` is write-only; pass `""` to clear it.

### `delete_profile` `{id}` → Settings

### `test_profile` `{id}` → `{"ok": bool, "message": str, "kind": str, "models": [str…]}`
Network call on a worker thread; other requests keep being served meanwhile.

### `export` `{destination, what: "images"|"translation_txt"|"source_txt"}` → `{"files": [path…]}`
`images` copies every existing result image into `destination` (created if needed),
keeping file names. Text exports write one file into `destination`.

### `shutdown` → `{"ok": true}` then the process exits.

## Events

| event | fields |
|---|---|
| `prepare_progress` | `module` (`detector`/`ocr`/`inpainter`/`translator`/`packages`), `phase` (e.g. `downloading`, `installing`, `loading`), `message` (short English), `done`?, `total`? (bytes) |
| `stage_progress` | `stage` (`detect`/`ocr`/`translate`/`inpaint`), `done`, `total` (pages) |
| `page_done` | `page` (name), `revision` — result image written; fetch with `get_page` |
| `run_finished` | `cancelled`: bool, `failed_pages`: [name…] |
| `error` | `message`, `page`? — non-fatal problem during a run (e.g. translation failed for a page) |
| `log` | `level`, `message` — optional, for a diagnostics window |

## Error codes

`bad_request`, `not_found`, `busy`, `no_project`, `translator_not_ready`, `internal`.

## Engine notes

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
