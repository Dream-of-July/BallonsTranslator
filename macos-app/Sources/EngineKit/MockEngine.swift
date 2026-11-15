import CoreGraphics
import CoreText
import Foundation
import ImageIO

/// In-process stand-in for the Python engine: same protocol JSON, placeholder pages drawn
/// at runtime. Used by previews, tests, snapshots and the `--mock` launch flag.
public actor MockEngine: Engine {
    public struct Options: Sendable {
        public var pageCount = 6
        /// Delay per page per stage, and per prepare step.
        public var stepDelay: Duration = .milliseconds(180)
        /// `false` makes the first run emit `prepare_progress` first.
        public var modelsReady = true
        /// Pages that report a translation error.
        public var failingPages: Set<String> = []
        /// Pages already translated when the project opens.
        public var translatedPages = 0
        /// `false` behaves like an engine without the `keep_original` edit.
        public var supportsKeepOriginal = true
        /// `false` behaves like an engine without `probe_vision` and the `enhance` setting.
        public var supportsVision = true

        public init() {}
    }

    public nonisolated let events: AsyncStream<EngineEvent>
    private let sink: AsyncStream<EngineEvent>.Continuation
    private var options: Options
    private let directory: URL

    private var project: Project?
    private var settings: EngineSettings
    private var runTask: Task<Void, Never>?
    private var cancelRequested = false
    private var log: [String] = ["mock engine"]

    public init(options: Options = Options()) {
        self.options = options
        (events, sink) = AsyncStream.makeStream(of: EngineEvent.self)
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MangaTranslatorMock-\(UUID().uuidString)")
        settings = EngineSettings(
            sourceLanguage: "日本語",
            targetLanguage: "简体中文",
            languages: LanguageOptions(source: ["日本語", "English", "한국어"],
                                       target: ["简体中文", "繁體中文", "English"]),
            translator: TranslatorSelection(kind: .google, module: "google"),
            // One profile per `vision` state: measured no, measured yes, not measured yet.
            profiles: [
                Profile(id: "p1", name: "DeepSeek", baseURL: "https://api.deepseek.com/v1", model: "deepseek-chat",
                        modelOptions: ["deepseek-chat", "deepseek-reasoner"], requireAPIKey: true, hasAPIKey: false,
                        vision: options.supportsVision ? .no : .unknown),
                Profile(id: "p2", name: "OpenAI", baseURL: "https://api.openai.com/v1", model: "gpt-4o-mini",
                        modelOptions: ["gpt-4o-mini", "gpt-4o"], requireAPIKey: true, hasAPIKey: true,
                        vision: options.supportsVision ? .yes : .unknown),
                Profile(id: "p3", name: "Ollama", baseURL: "http://localhost:11434/v1", model: "qwen2.5vl",
                        modelOptions: [], requireAPIKey: false, hasAPIKey: false),
            ],
            providerTemplates: ["OpenAI", "DeepSeek", "Ollama"],
            output: OutputSettings(format: ".png", quality: 95)
        )
    }

    public func start() async throws {}

    public func stop() async {
        runTask?.cancel()
        runTask = nil
    }

    public func diagnostics() -> [String] { log }

    public func call(_ method: String, params: JSONValue?) async throws -> JSONValue {
        log.append("→ \(method)")
        switch method {
        case "hello":
            return try JSONValue(encoding: Hello(protocolVersion: engineProtocolVersion, engineVersion: "mock",
                                                 device: "mps", torchInstalled: true, modelsReady: options.modelsReady,
                                                 capabilities: capabilities))
        case "open_project":
            return try openProject(path: try string(params, "path"))
        case "get_page":
            return try JSONValue(encoding: try page(try string(params, "name")).1)
        case "run":
            return try startRun(params)
        case "cancel":
            cancelRequested = true
            return ["cancelling": true]
        case "update_blocks":
            return try updateBlocks(params)
        case "delete_blocks":
            return try deleteBlocks(params)
        case "list_fonts":
            return try JSONValue(encoding: FontList(fonts: [
                FontInfo(family: "PingFang SC", display: "PingFang SC", source: "system"),
                FontInfo(family: "Hiragino Sans", display: "Hiragino Sans", source: "system"),
                FontInfo(family: "Helvetica Neue", display: "Helvetica Neue", source: "system"),
                FontInfo(family: "Arial Rounded MT Bold", display: "Arial Rounded MT Bold", source: "system"),
                FontInfo(family: "LXGW WenKai", display: "LXGW WenKai", source: "custom"),
            ], default: "PingFang SC"))
        case "get_settings":
            return try JSONValue(encoding: settings)
        case "set_settings":
            return try setSettings(params)
        case "create_profile":
            let template = try string(params, "template")
            guard settings.providerTemplates.contains(template) else { throw error(.badRequest, "Unknown template.") }
            let local = template == "Ollama"
            let profile = Profile(
                id: "p\(UUID().uuidString.prefix(6))", name: template,
                baseURL: local ? "http://localhost:11434/v1" : "https://api.\(template.lowercased()).com/v1",
                model: "", modelOptions: [], requireAPIKey: !local, hasAPIKey: false)
            settings.profiles.append(profile)
            return try JSONValue(encoding: profile)
        case "update_profile":
            let index = try profileIndex(params)
            if let name = params?["name"]?.stringValue { settings.profiles[index].name = name }
            if let baseURL = params?["base_url"]?.stringValue { settings.profiles[index].baseURL = baseURL }
            if let model = params?["model"]?.stringValue { settings.profiles[index].model = model }
            // A different endpoint or model has to be measured again.
            if params?["base_url"]?.stringValue != nil || params?["model"]?.stringValue != nil { settings.profiles[index].vision = .unknown }
            if let key = params?["api_key"]?.stringValue { settings.profiles[index].hasAPIKey = !key.isEmpty }
            return try JSONValue(encoding: settings.profiles[index])
        case "delete_profile":
            let removed = settings.profiles.remove(at: try profileIndex(params))
            if settings.translator.profileID == removed.id {
                settings.translator = TranslatorSelection(kind: .google, module: "google")
            }
            if settings.enhance.profileID == removed.id { settings.enhance.profileID = nil }
            return try JSONValue(encoding: settings)
        case "test_profile":
            let index = try profileIndex(params)
            try? await Task.sleep(for: options.stepDelay * 3)
            let profile = settings.profiles[index]
            guard profile.hasAPIKey || !profile.requireAPIKey else {
                return try JSONValue(encoding: ProfileTestResult(ok: false, message: "API key is missing.", kind: "auth", models: []))
            }
            let models = profile.modelOptions.isEmpty ? ["model-small", "model-large"] : profile.modelOptions
            if let index = settings.profiles.firstIndex(where: { $0.id == profile.id }) {
                settings.profiles[index].modelOptions = models
            }
            return try JSONValue(encoding: ProfileTestResult(ok: true, message: "Connected.", kind: "ok", models: models))
        case "probe_vision":
            guard options.supportsVision else { throw error(.badRequest, "Unknown method \(method).") }
            guard let id = params?["profile_id"]?.stringValue,
                  let index = settings.profiles.firstIndex(where: { $0.id == id }) else {
                throw error(.notFound, "No such profile.")
            }
            try? await Task.sleep(for: options.stepDelay * 3)
            let profile = settings.profiles[index]
            guard profile.hasAPIKey || !profile.requireAPIKey else {
                return try JSONValue(encoding: ProfileTestResult(ok: false, message: "API key is missing.", kind: "auth", models: []))
            }
            // Stand-in for reading the digits back: model names that usually take images.
            let model = profile.model.lowercased()
            let vision: VisionSupport = ["vl", "4o", "vision", "gemini"].contains(where: model.contains) ? .yes : .no
            settings.profiles[index].vision = vision
            return ["vision": .string(vision.rawValue)]
        case "export":
            return try export(params)
        case "shutdown":
            await stop()
            return ["ok": true]
        default:
            throw error(.badRequest, "Unknown method \(method).")
        }
    }

    // MARK: - Methods

    private func openProject(path: String) throws -> JSONValue {
        guard !path.isEmpty, !path.hasSuffix("/missing") else { throw error(.notFound, "Folder does not exist.") }
        guard runTask == nil else { throw error(.busy, "A translation is running.") }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("result"),
                                                withIntermediateDirectories: true)
        var pages: [Page] = []
        for number in 0..<options.pageCount {
            let name = String(format: "%03d.png", number + 1)
            let source = directory.appendingPathComponent(name)
            try MockArt.write(MockArt.page(number: number, blocks: nil), to: source)
            var page = Page(name: name, source: source.path, width: Int(MockArt.size.width), height: Int(MockArt.size.height))
            if number < options.translatedPages {
                page.blocks = MockArt.blocks(number: number, english: targetIsEnglish)
                try render(&page, number: number)
            }
            pages.append(page)
        }
        let project = Project(path: path, name: URL(fileURLWithPath: path).lastPathComponent, pages: pages)
        self.project = project
        return try JSONValue(encoding: project)
    }

    private func startRun(_ params: JSONValue?) throws -> JSONValue {
        guard let project else { throw error(.noProject, "No project is open.") }
        guard runTask == nil else { throw error(.busy, "A translation is running.") }
        if settings.translator.kind == .llm {
            guard let profile = settings.profiles.first(where: { $0.id == settings.translator.profileID }) else {
                throw error(.translatorNotReady, "Choose a translation service.", data: ["missing": "model"])
            }
            if profile.requireAPIKey && !profile.hasAPIKey {
                throw error(.translatorNotReady, "\(profile.name) needs an API key.", data: ["missing": "api_key"])
            }
            if profile.model.isEmpty {
                throw error(.translatorNotReady, "\(profile.name) needs a model.", data: ["missing": "model"])
            }
        }
        var names = params?["pages"]?.arrayValue?.compactMap(\.stringValue) ?? project.pages.map(\.name)
        if params?["mode"]?.stringValue == RunMode.continue.rawValue {
            let done = Set(project.pages.filter(\.done).map(\.name))
            names.removeAll(where: done.contains)
        }
        cancelRequested = false
        runTask = Task { await performRun(names) }
        return ["started": true]
    }

    private func performRun(_ names: [String]) async {
        var failed: [String] = []
        if !options.modelsReady {
            let total = 180_000_000.0
            for step in 1...6 where !cancelRequested {
                sink.yield(.prepareProgress(PrepareProgress(
                    module: "ocr", phase: "downloading", message: "Downloading text recognition model",
                    done: total * Double(step) / 6, total: total)))
                try? await Task.sleep(for: options.stepDelay)
            }
            sink.yield(.prepareProgress(PrepareProgress(module: "ocr", phase: "loading", message: "Loading models")))
            try? await Task.sleep(for: options.stepDelay)
            options.modelsReady = true
        }
        let enhance = options.supportsVision && settings.enhance.enabled
        stages: for stage in ["detect", "ocr", "translate", "inpaint"] + (enhance ? ["enhance"] : []) {
            for (offset, name) in names.enumerated() {
                if cancelRequested || Task.isCancelled { break stages }
                try? await Task.sleep(for: options.stepDelay)
                if stage == "translate", options.failingPages.contains(name) {
                    failed.append(name)
                    sink.yield(.error(message: "Translation failed for \(name).", page: name))
                }
                sink.yield(.stageProgress(stage: stage, done: offset + 1, total: names.count))
                if stage == "inpaint", !failed.contains(name), let (index, _) = try? page(name) {
                    project?.pages[index].blocks = MockArt.blocks(number: index, english: targetIsEnglish)
                    if var page = project?.pages[index], (try? render(&page, number: index)) != nil {
                        project?.pages[index] = page
                        sink.yield(.pageDone(page: name, revision: page.revision))
                    }
                }
            }
        }
        runTask = nil
        sink.yield(.runFinished(cancelled: cancelRequested, failedPages: failed))
    }

    private func updateBlocks(_ params: JSONValue?) throws -> JSONValue {
        guard runTask == nil else { throw error(.busy, "A translation is running.") }
        let (index, current) = try page(try string(params, "page"))
        guard let edits = try? params?["edits"]?.decode([BlockEdit].self) else {
            throw error(.badRequest, "Missing edits.")
        }
        var page = current
        let base = MockArt.blocks(number: index, english: targetIsEnglish)
        for edit in edits {
            guard let position = page.blocks.firstIndex(where: { $0.index == edit.index }) else {
                throw error(.badRequest, "No block \(edit.index).")
            }
            guard edit.keepOriginal == nil || options.supportsKeepOriginal else {
                throw error(.badRequest, "Invalid edit.")
            }
            edit.apply(to: &page.blocks[position])
            let original = base.first { $0.index == edit.index }
            if edit.keepOriginal != true, edit.keepOriginal != nil || edit.sourceText != nil || edit.translation != nil {
                // Stand-in for the engine's checks: a withheld block passes once its text is changed.
                let block = page.blocks[position]
                let unchanged = block.sourceText == original?.sourceText && block.translation == original?.translation
                page.blocks[position].reviewReason = unchanged ? original?.reviewReason : nil
                if original?.reviewReason == Block.skipped, !unchanged, block.translation.isEmpty {
                    // Stand-in for translating the source text the user typed in.
                    page.blocks[position].translation = block.sourceText
                }
            }
            if params?["autofit"]?.boolValue == true, let original {
                // Stand-in for the engine's layout: back to the bubble, sized to fill it.
                page.blocks[position].rect = original.rect
                page.blocks[position].fontSize = original.fontSize
            }
        }
        try render(&page, number: index)
        project?.pages[index] = page
        return try JSONValue(encoding: page)
    }

    private func deleteBlocks(_ params: JSONValue?) throws -> JSONValue {
        guard runTask == nil else { throw error(.busy, "A translation is running.") }
        let (index, current) = try page(try string(params, "page"))
        let indexes = Set(params?["indexes"]?.arrayValue?.compactMap(\.intValue) ?? [])
        var page = current
        page.blocks.removeAll { indexes.contains($0.index) }
        try render(&page, number: index)
        project?.pages[index] = page
        return try JSONValue(encoding: page)
    }

    private func setSettings(_ params: JSONValue?) throws -> JSONValue {
        if let source = params?["source_language"]?.stringValue { settings.sourceLanguage = source }
        if let target = params?["target_language"]?.stringValue { settings.targetLanguage = target }
        if let translator = params?["translator"] {
            switch translator["kind"]?.stringValue {
            case "google":
                settings.translator = TranslatorSelection(kind: .google, module: "google")
            case "llm":
                guard let id = translator["profile_id"]?.stringValue, settings.profiles.contains(where: { $0.id == id }) else {
                    throw error(.notFound, "No such profile.")
                }
                settings.translator = TranslatorSelection(kind: .llm, module: "LLM_API_Translator", profileID: id)
            default:
                throw error(.badRequest, "Unknown translator.")
            }
        }
        if let output = try? params?["output"]?.decode(OutputSettings.self) { settings.output = output }
        if let enhance = params?["enhance"], options.supportsVision {
            if let enabled = enhance["enabled"]?.boolValue { settings.enhance.enabled = enabled }
            switch enhance["profile_id"] {
            case .none: break
            case .some(.null): settings.enhance.profileID = nil
            case .some(let value):
                guard let id = value.stringValue, settings.profiles.contains(where: { $0.id == id }) else {
                    throw error(.notFound, "No such profile.")
                }
                settings.enhance.profileID = id
            }
        }
        return try JSONValue(encoding: settings)
    }

    private var capabilities: [String] {
        (options.supportsKeepOriginal ? [Hello.keepOriginal] : [])
            + (options.supportsVision ? [Hello.visionProbe, Hello.enhance] : [])
    }

    private func export(_ params: JSONValue?) throws -> JSONValue {
        guard let project else { throw error(.noProject, "No project is open.") }
        let destination = URL(fileURLWithPath: try string(params, "destination"))
        do {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            var files: [String] = []
            for page in project.pages {
                guard let result = page.result else { continue }
                let target = destination.appendingPathComponent(page.name)
                try? FileManager.default.removeItem(at: target)
                try FileManager.default.copyItem(at: URL(fileURLWithPath: result), to: target)
                files.append(target.path)
            }
            return ["files": .array(files.map(JSONValue.string))]
        } catch {
            throw self.error(.internal, "Could not write to the folder.")
        }
    }

    // MARK: - Helpers

    private var targetIsEnglish: Bool { settings.targetLanguage == "English" }

    private func render(_ page: inout Page, number: Int) throws {
        let result = directory.appendingPathComponent("result").appendingPathComponent(page.name)
        try MockArt.write(MockArt.page(number: number, blocks: page.blocks), to: result)
        page.result = result.path
        page.done = true
        page.revision += 1
    }

    private func page(_ name: String) throws -> (Int, Page) {
        guard let project else { throw error(.noProject, "No project is open.") }
        guard let index = project.pages.firstIndex(where: { $0.name == name }) else {
            throw error(.notFound, "No page named \(name).")
        }
        return (index, project.pages[index])
    }

    private func profileIndex(_ params: JSONValue?) throws -> Int {
        let id = try string(params, "id")
        guard let index = settings.profiles.firstIndex(where: { $0.id == id }) else {
            throw error(.notFound, "No such profile.")
        }
        return index
    }

    private func string(_ params: JSONValue?, _ key: String) throws -> String {
        guard let value = params?[key]?.stringValue else { throw error(.badRequest, "Missing \(key).") }
        return value
    }

    private func error(_ code: ErrorCode, _ message: String, data: JSONValue? = nil) -> EngineError {
        .remote(code: code.rawValue, message: message, data: data)
    }
}

/// Placeholder manga pages: grey panels, blob figures and speech bubbles. No real artwork.
enum MockArt {
    static let size = CGSize(width: 800, height: 1200)
    private static let fontLock = NSLock()

    private static let lines: [(source: String, chinese: String, english: String)] = [
        ("おはよう!", "早上好!", "Good morning!"),
        ("今日はいい天気だね", "今天天气真好啊", "Nice weather today."),
        ("さあ、行こう!", "好,出发吧!", "Come on, let's go!"),
        ("ちょっと待って", "等一下", "Wait a second."),
        ("本当に?", "真的吗?", "Really?"),
    ]

    private static let bubbleCount = 3

    /// Text boxes of a page, already "translated": three bubbles, and on the second, fourth and
    /// sixth page the mix a real run leaves behind — withheld blocks, sound effects, a signature.
    static func blocks(number: Int, english: Bool) -> [Block] {
        let shift = CGFloat((number * 37) % 60)
        let rects = [
            CGRect(x: 90 + shift, y: 120, width: 220, height: 100),
            CGRect(x: 470, y: 480 + shift, width: 230, height: 110),
            CGRect(x: 150 + shift, y: 890, width: 260, height: 100),
        ]
        var blocks = rects.enumerated().map { index, rect in
            let line = lines[(number + index) % lines.count]
            return Block(index: index, rect: rect, angle: index == 1 ? -6 : 0, sourceText: line.source,
                         translation: english ? line.english : line.chinese, vertical: false,
                         fontFamily: "PingFang SC", fontSize: 34, color: RGB(r: 20, g: 20, b: 20),
                         alignment: .center, bold: false, italic: false)
        }
        func withhold(_ index: Int, _ reason: String, translation: String) {
            blocks[index].reviewReason = reason
            blocks[index].translation = translation
            blocks[index].angle = 0
            // The detected box hugs the line of original text at the top of the bubble.
            blocks[index].sourceRect = CGRect(x: blocks[index].rect.minX + 8, y: blocks[index].rect.minY - 4,
                                              width: blocks[index].rect.width - 16, height: 52)
        }
        switch number {
        case 1:
            withhold(1, "unreadable", translation: "")
            let effect = CGRect(x: 520, y: 165, width: 190, height: 64)
            blocks.append(Block(index: 3, rect: effect, sourceText: "ドン!", fontFamily: "PingFang SC", fontSize: 46,
                                color: RGB(r: 20, g: 20, b: 20), bold: true, sourceRect: effect, reviewReason: "untranslated"))
            let signature = CGRect(x: 70, y: 1106, width: 150, height: 34)
            blocks.append(Block(index: 4, rect: signature, sourceText: "たなか 画", fontFamily: "PingFang SC", fontSize: 22,
                                color: RGB(r: 20, g: 20, b: 20), sourceRect: signature, reviewReason: Block.protected))
        case 3:
            withhold(0, "overflow", translation: english
                ? "Hold on, wait for me, I still have to find my umbrella and my keys!"
                : "等一下等一下,我还得先找到我的雨伞和钥匙才能出门!")
            withhold(2, "geometry", translation: blocks[2].translation)
            blocks.append(skipped(index: 3, CGRect(x: 560, y: 820, width: 150, height: 60), "ゴゴゴ"))
        case 5:
            blocks.append(skipped(index: 3, CGRect(x: 600, y: 300, width: 120, height: 50), "ザワ"))
        default:
            break
        }
        return blocks
    }

    /// Unreadable text drawn into the artwork, outside every bubble: left as it is.
    private static func skipped(index: Int, _ rect: CGRect, _ text: String) -> Block {
        Block(index: index, rect: rect, sourceText: text, fontFamily: "PingFang SC", fontSize: 40,
              color: RGB(r: 20, g: 20, b: 20), bold: true, sourceRect: rect, reviewReason: Block.skipped)
    }

    /// Draws page `number`. `blocks == nil` gives the untranslated source page.
    static func page(number: Int, blocks: [Block]?) -> CGImage? {
        guard let context = CGContext(
            data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        // Work in image coordinates (origin top-left) for shapes.
        context.translateBy(x: 0, y: size.height)
        context.scaleBy(x: 1, y: -1)
        context.setFillColor(gray: 1, alpha: 1)
        context.fill(CGRect(origin: .zero, size: size))

        let panels = [
            CGRect(x: 40, y: 40, width: 720, height: 340),
            CGRect(x: 40, y: 410, width: 350, height: 360),
            CGRect(x: 410, y: 410, width: 350, height: 360),
            CGRect(x: 40, y: 800, width: 720, height: 360),
        ]
        context.setLineWidth(5)
        context.setStrokeColor(gray: 0.1, alpha: 1)
        for (index, panel) in panels.enumerated() {
            context.setFillColor(gray: 0.86 - 0.05 * CGFloat((index + number) % 3), alpha: 1)
            context.fill(panel)
            // A head-and-shoulders blob per panel.
            context.saveGState()
            context.clip(to: panel)
            context.setFillColor(gray: 0.55, alpha: 1)
            let head = CGRect(x: panel.maxX - 190, y: panel.midY - 70, width: 110, height: 110)
            context.fillEllipse(in: head)
            context.fillEllipse(in: CGRect(x: head.midX - 120, y: head.maxY - 10, width: 240, height: 260))
            context.restoreGState()
            context.stroke(panel)
        }

        let base = MockArt.blocks(number: number, english: false)
        context.setLineWidth(4)
        for bubble in base.prefix(bubbleCount) {
            let ellipse = bubble.rect.insetBy(dx: -34, dy: -30)
            context.setFillColor(gray: 1, alpha: 1)
            context.fillEllipse(in: ellipse)
            context.strokeEllipse(in: ellipse)
        }
        for block in blocks ?? base {
            // A withheld block keeps its original text on the translated page.
            let original = blocks == nil || block.isWithheld
            draw(original ? block.sourceText : block.translation, block: block,
                 family: original ? "Hiragino Sans" : block.fontFamily, in: context)
        }
        return context.makeImage()
    }

    private static func draw(_ text: String, block: Block, family: String, in context: CGContext) {
        var traits: CTFontSymbolicTraits = []
        if block.bold { traits.insert(.traitBold) }
        if block.italic { traits.insert(.traitItalic) }
        // One lookup at a time: concurrent by-name lookups from several mock engines
        // (parallel tests) can leave CoreText waiting on the font daemon forever.
        let font = fontLock.withLock {
            let font = CTFontCreateWithName(family as CFString, block.fontSize, nil)
            return traits.isEmpty ? font : CTFontCreateCopyWithSymbolicTraits(font, 0, nil, traits, traits) ?? font
        }
        var alignment: CTTextAlignment = switch block.alignment {
        case .left: .left
        case .center: .center
        case .right: .right
        }
        let paragraph = withUnsafePointer(to: &alignment) { pointer in
            var setting = CTParagraphStyleSetting(spec: .alignment, valueSize: MemoryLayout<CTTextAlignment>.size, value: pointer)
            return CTParagraphStyleCreate(&setting, 1)
        }
        let color = CGColor(red: CGFloat(block.color.r) / 255, green: CGFloat(block.color.g) / 255,
                            blue: CGFloat(block.color.b) / 255, alpha: 1)
        let attributed = NSAttributedString(string: text, attributes: [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
            NSAttributedString.Key(kCTParagraphStyleAttributeName as String): paragraph,
        ])
        let box = CGRect(x: -block.rect.width / 2, y: -block.rect.height / 2,
                         width: block.rect.width, height: block.rect.height)
        let frame = CTFramesetterCreateFrame(CTFramesetterCreateWithAttributedString(attributed),
                                             CFRange(location: 0, length: 0), CGPath(rect: box, transform: nil), nil)
        context.saveGState()
        context.translateBy(x: block.rect.midX, y: block.rect.midY)
        context.rotate(by: block.angle * .pi / 180)
        // CoreText lays out bottom-up; undo the page flip around the box centre.
        context.scaleBy(x: 1, y: -1)
        CTFrameDraw(frame, context)
        context.restoreGState()
    }

    static func write(_ image: CGImage?, to url: URL) throws {
        guard let image,
              let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            throw EngineError.remote(code: ErrorCode.internal.rawValue, message: "Could not draw the page.", data: nil)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw EngineError.remote(code: ErrorCode.internal.rawValue, message: "Could not save the page.", data: nil)
        }
    }
}
