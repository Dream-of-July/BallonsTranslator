import CoreGraphics
import Foundation
import Testing
@testable import EngineKit

private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
    try JSONDecoder().decode(T.self, from: Data(json.utf8))
}

private func encodeToValue(_ value: some Encodable) throws -> JSONValue {
    try JSONValue(encoding: value)
}

@Suite struct ProtocolTests {
    @Test func helloDecodes() throws {
        let hello = try decode(Hello.self, """
        {"protocol": 1, "engine_version": "1.5.21", "device": "mps", "torch_installed": true, "models_ready": false}
        """)
        #expect(hello == Hello(protocolVersion: 1, engineVersion: "1.5.21", device: "mps", torchInstalled: true, modelsReady: false))
        let noDevice = try decode(Hello.self, """
        {"protocol": 1, "engine_version": "1.5.21", "device": null, "torch_installed": false, "models_ready": false}
        """)
        #expect(noDevice.device == nil)
        #expect(noDevice.capabilities.isEmpty)
        let capable = try decode(Hello.self, """
        {"protocol": 1, "engine_version": "x", "device": null, "torch_installed": true, "models_ready": true,
         "capabilities": ["keep_original"]}
        """)
        #expect(capable.capabilities == [Hello.keepOriginal])
    }

    @Test func projectPageAndBlockDecode() throws {
        let project = try decode(Project.self, """
        {"path": "/m/chapter1", "name": "chapter1", "pages": [
          {"name": "001.png", "source": "/abs/001.png", "width": 800, "height": 1200,
           "result": "/abs/result/001.png", "inpainted": null, "done": true, "revision": 3,
           "blocks": [{"index": 0, "rect": [10, 20.5, 300, 120], "angle": -6,
                       "source_text": "おはよう", "translation": "早上好", "vertical": true,
                       "font_family": "PingFang SC", "font_size": 34.5, "color": [20, 30, 40],
                       "alignment": "center", "bold": false, "italic": true}]},
          {"name": "002.png", "source": "/abs/002.png", "width": 800, "height": 1200,
           "result": null, "inpainted": null, "done": false, "revision": 0, "blocks": []}
        ]}
        """)
        #expect(project.name == "chapter1")
        #expect(project.pages.count == 2)
        let page = project.pages[0]
        #expect(page.result == "/abs/result/001.png")
        #expect(page.inpainted == nil)
        #expect(page.revision == 3)
        let block = try #require(page.blocks.first)
        #expect(block.rect == CGRect(x: 10, y: 20.5, width: 300, height: 120))
        #expect(block.angle == -6)
        #expect(block.sourceText == "おはよう")
        #expect(block.vertical)
        #expect(block.fontSize == 34.5)
        #expect(block.color == RGB(r: 20, g: 30, b: 40))
        #expect(block.alignment == .center)
        #expect(block.italic && !block.bold)
        #expect(project.pages[1].result == nil && !project.pages[1].done)

        // Round trip keeps the wire shape.
        let wire = try encodeToValue(block)
        #expect(wire["rect"] == [10, 20.5, 300, 120])
        #expect(wire["font_family"] == "PingFang SC")
        #expect(wire["color"] == [20, 30, 40])
        #expect(try wire.decode(Block.self) == block)
    }

    @Test func settingsDecode() throws {
        let settings = try decode(EngineSettings.self, """
        {"source_language": "日本語", "target_language": "简体中文",
         "languages": {"source": ["日本語"], "target": ["简体中文", "English"]},
         "translator": {"kind": "llm", "module": "LLM_API_Translator", "profile_id": "abc"},
         "profiles": [{"id": "abc", "name": "DeepSeek", "base_url": "https://api.deepseek.com/v1",
                       "model": "deepseek-chat", "model_options": ["deepseek-chat"],
                       "require_api_key": true, "has_api_key": false}],
         "provider_templates": ["OpenAI", "DeepSeek"],
         "output": {"format": ".png", "quality": 95}}
        """)
        #expect(settings.sourceLanguage == "日本語")
        #expect(settings.languages.target == ["简体中文", "English"])
        #expect(settings.translator == TranslatorSelection(kind: .llm, module: "LLM_API_Translator", profileID: "abc"))
        #expect(settings.profiles.first?.baseURL == "https://api.deepseek.com/v1")
        #expect(settings.profiles.first?.requireAPIKey == true)
        #expect(settings.profiles.first?.hasAPIKey == false)
        #expect(settings.output == OutputSettings(format: ".png", quality: 95))

        let google = try decode(TranslatorSelection.self, #"{"kind": "google", "module": "google", "profile_id": null}"#)
        #expect(google.kind == .google && google.profileID == nil)
        let unknown = try decode(TranslatorSelection.self, #"{"kind": "sakura", "module": "x", "profile_id": null}"#)
        #expect(unknown.kind == .other)
    }

    @Test func visionAndEnhanceFieldsAreOptional() throws {
        // An engine from before `vision_probe` / `enhance` sends neither field.
        let old = try decode(EngineSettings.self, """
        {"source_language": "日本語", "target_language": "简体中文",
         "languages": {"source": ["日本語"], "target": ["简体中文"]},
         "translator": {"kind": "google", "module": "google", "profile_id": null},
         "profiles": [{"id": "a", "name": "A", "base_url": "u", "model": "m", "model_options": [],
                       "require_api_key": true, "has_api_key": false}],
         "provider_templates": [], "output": {"format": ".png", "quality": 95}}
        """)
        #expect(old.profiles.first?.vision == .unknown)
        #expect(old.enhance == EnhanceSettings(enabled: false, profileID: nil))

        let current = try decode(EngineSettings.self, """
        {"source_language": "日本語", "target_language": "简体中文",
         "languages": {"source": ["日本語"], "target": ["简体中文"]},
         "translator": {"kind": "llm", "module": "LLM_API_Translator", "profile_id": "a"},
         "profiles": [{"id": "a", "name": "A", "base_url": "u", "model": "m", "model_options": [],
                       "require_api_key": true, "has_api_key": true, "vision": "yes"},
                      {"id": "b", "name": "B", "base_url": "u", "model": "m", "model_options": [],
                       "require_api_key": true, "has_api_key": true, "vision": "maybe"}],
         "provider_templates": [], "output": {"format": ".png", "quality": 95},
         "enhance": {"enabled": true, "profile_id": null}}
        """)
        #expect(current.profiles.map(\.vision) == [.yes, .unknown])
        #expect(current.enhance == EnhanceSettings(enabled: true, profileID: nil))
        #expect(current.enhanceIsOn)
        // Round trip through the mock's encoding keeps the fields.
        #expect(try encodeToValue(current).decode(EngineSettings.self) == current)

        let hello = try decode(Hello.self, """
        {"protocol": 1, "engine_version": "x", "device": null, "torch_installed": true, "models_ready": true,
         "capabilities": ["keep_original", "vision_probe", "enhance"]}
        """)
        #expect(hello.capabilities.contains(Hello.visionProbe) && hello.capabilities.contains(Hello.enhance))
    }

    @Test func visionProbeResultDecodes() throws {
        #expect(try decode(VisionProbeResult.self, #"{"vision": "no"}"#) == .measured(.no))
        #expect(try decode(VisionProbeResult.self, #"{"ok": false, "message": "Connection failed.", "kind": "network", "models": []}"#)
            == .failed(ProfileTestResult(ok: false, message: "Connection failed.", kind: "network", models: [])))
    }

    @Test func enhanceAvailabilityFollowsTheEffectiveProfile() {
        var settings = EngineSettings(
            sourceLanguage: "日本語", targetLanguage: "简体中文", languages: LanguageOptions(source: [], target: []),
            translator: TranslatorSelection(kind: .google, module: "google"),
            profiles: [Profile(id: "no", name: "N", baseURL: "", model: "", vision: .no),
                       Profile(id: "yes", name: "Y", baseURL: "", model: "", vision: .yes),
                       Profile(id: "new", name: "U", baseURL: "", model: "")],
            providerTemplates: [], output: OutputSettings(format: ".png", quality: 95),
            enhance: EnhanceSettings(enabled: true))
        // Google translates and nothing is chosen: no service to re-check with.
        #expect(settings.enhanceProfile == nil)
        #expect(settings.enhanceAvailability == .noProfile)
        #expect(!settings.enhanceIsOn)

        settings.enhance.profileID = "yes"
        #expect(settings.enhanceAvailability == .available)
        #expect(settings.enhanceIsOn)

        // "Same as translation" follows the translator.
        settings.enhance.profileID = nil
        settings.translator = TranslatorSelection(kind: .llm, module: "LLM_API_Translator", profileID: "no")
        #expect(settings.enhanceAvailability == .visionUnsupported)
        #expect(!settings.enhanceIsOn)
        settings.translator.profileID = "new"
        #expect(settings.enhanceAvailability == .visionUnknown(profileID: "new"))

        // A chosen profile wins over the translator's, and a deleted one leaves nothing.
        settings.enhance.profileID = "yes"
        #expect(settings.enhanceProfile?.id == "yes")
        settings.enhance.profileID = "gone"
        #expect(settings.enhanceAvailability == .noProfile)
    }

    @Test func eventsDecode() throws {
        #expect(try decode(EngineEvent.self, #"{"event": "stage_progress", "stage": "enhance", "done": 1, "total": 4}"#)
            == .stageProgress(stage: "enhance", done: 1, total: 4))
        #expect(try decode(EngineEvent.self, #"{"event": "stage_progress", "stage": "ocr", "done": 3, "total": 12}"#)
            == .stageProgress(stage: "ocr", done: 3, total: 12))
        #expect(try decode(EngineEvent.self, #"{"event": "prepare_progress", "module": "ocr", "phase": "downloading", "message": "Downloading", "done": 10, "total": 200}"#)
            == .prepareProgress(PrepareProgress(module: "ocr", phase: "downloading", message: "Downloading", done: 10, total: 200)))
        #expect(try decode(EngineEvent.self, #"{"event": "prepare_progress", "module": "packages", "phase": "installing", "message": "Installing"}"#)
            == .prepareProgress(PrepareProgress(module: "packages", phase: "installing", message: "Installing")))
        #expect(try decode(EngineEvent.self, #"{"event": "page_done", "page": "001.png", "revision": 2}"#)
            == .pageDone(page: "001.png", revision: 2))
        #expect(try decode(EngineEvent.self, #"{"event": "run_finished", "cancelled": true, "failed_pages": ["002.png"]}"#)
            == .runFinished(cancelled: true, failedPages: ["002.png"]))
        #expect(try decode(EngineEvent.self, #"{"event": "error", "message": "Translation failed."}"#)
            == .error(message: "Translation failed.", page: nil))
        #expect(try decode(EngineEvent.self, #"{"event": "log", "level": "warning", "message": "slow"}"#)
            == .log(level: "warning", message: "slow"))
        #expect(throws: DecodingError.self) {
            try decode(EngineEvent.self, #"{"event": "something_new"}"#)
        }
    }

    @Test func requestsEncode() throws {
        let edit = BlockEdit(index: 2, rect: CGRect(x: 1, y: 2, width: 30, height: 40), fontSize: 28, bold: true)
        #expect(try encodeToValue(edit) == ["index": 2, "rect": [1, 2, 30, 40], "font_size": 28, "bold": true])
        #expect(try encodeToValue(edit).decode(BlockEdit.self) == edit)
        let correction = BlockEdit(index: 1, translation: "b", sourceText: "a", keepOriginal: false)
        #expect(try encodeToValue(correction) == ["index": 1, "source_text": "a", "translation": "b", "keep_original": false])
        #expect(try encodeToValue(correction).decode(BlockEdit.self) == correction)

        let patch = SettingsPatch(targetLanguage: "English", translator: .llm(profileID: "abc"))
        #expect(try encodeToValue(patch) == ["target_language": "English", "translator": ["kind": "llm", "profile_id": "abc"]])
        #expect(try encodeToValue(SettingsPatch(translator: .google)) == ["translator": ["kind": "google"]])
        #expect(try encodeToValue(ProfilePatch(baseURL: "http://x", apiKey: "")) == ["base_url": "http://x", "api_key": ""])
        #expect(try encodeToValue(SettingsPatch(enhance: EnhancePatch(enabled: true))) == ["enhance": ["enabled": true]])
        // `null` is sent on purpose: back to the translation service's profile.
        #expect(try encodeToValue(SettingsPatch(enhance: EnhancePatch(profileID: .some(nil)))) == ["enhance": ["profile_id": nil]])
        #expect(try encodeToValue(SettingsPatch(enhance: EnhancePatch(enabled: false, profileID: "p2")))
            == ["enhance": ["enabled": false, "profile_id": "p2"]])
        #expect(try encodeToValue(SettingsPatch(targetLanguage: "English")) == ["target_language": "English"])
    }

    @Test func errorCodesMatchTheContract() {
        #expect(ErrorCode.allCases.map(\.rawValue)
            == ["bad_request", "not_found", "busy", "no_project", "translator_not_ready", "internal"])
        let error = EngineError.remote(code: "translator_not_ready", message: "Needs a key.", data: ["missing": "api_key"])
        #expect(error.code == .translatorNotReady)
        #expect(error.missingTranslatorField == "api_key")
        #expect(error.localizedDescription == "Needs a key.")
    }

    @Test func editMergeAndInverse() {
        let block = Block(index: 0, rect: CGRect(x: 0, y: 0, width: 10, height: 10), translation: "a", fontSize: 20)
        let merged = BlockEdit(index: 0, translation: "b", fontSize: 22).merging(BlockEdit(index: 0, fontSize: 24, bold: true))
        #expect(merged == BlockEdit(index: 0, translation: "b", fontSize: 24, bold: true))
        #expect(merged.inverse(from: block) == BlockEdit(index: 0, translation: "a", fontSize: 20, bold: false))
        var edited = block
        merged.apply(to: &edited)
        #expect(edited.translation == "b" && edited.fontSize == 24 && edited.bold)

        var withheld = Block(index: 0, rect: block.rect, sourceRect: CGRect(x: 2, y: 2, width: 6, height: 6), reviewReason: "overflow")
        #expect(withheld.needsReview && withheld.visibleRect == withheld.sourceRect)
        let keep = BlockEdit(index: 0, keepOriginal: true)
        #expect(keep.inverse(from: withheld) == BlockEdit(index: 0, keepOriginal: false))
        keep.apply(to: &withheld)
        #expect(withheld.isWithheld && !withheld.needsReview)
        #expect(!block.isWithheld && block.visibleRect == block.rect)

        let skipped = Block(index: 1, rect: block.rect, sourceRect: block.rect, reviewReason: Block.skipped)
        #expect(skipped.isWithheld && skipped.isSkipped && !skipped.needsReview)
    }
}
