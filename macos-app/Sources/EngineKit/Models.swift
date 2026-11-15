import CoreGraphics
import Foundation

/// Version of `PROTOCOL.md` these models mirror.
public let engineProtocolVersion = 1

// MARK: - hello

public struct Hello: Codable, Equatable, Sendable {
    public var protocolVersion: Int
    public var engineVersion: String
    public var device: String?
    public var torchInstalled: Bool
    public var modelsReady: Bool
    /// Optional protocol additions this engine accepts; empty for engines that list none.
    public var capabilities: [String]

    /// Capability: `update_blocks` edits may carry `keep_original`.
    public static let keepOriginal = "keep_original"
    /// Capability: `probe_vision` exists and profiles carry a measured `vision`.
    public static let visionProbe = "vision_probe"
    /// Capability: settings carry `enhance` (vision re-check of hard blocks).
    public static let enhance = "enhance"

    enum CodingKeys: String, CodingKey {
        case protocolVersion = "protocol"
        case engineVersion = "engine_version"
        case device
        case torchInstalled = "torch_installed"
        case modelsReady = "models_ready"
        case capabilities
    }

    public init(protocolVersion: Int, engineVersion: String, device: String?, torchInstalled: Bool, modelsReady: Bool,
                capabilities: [String] = []) {
        self.protocolVersion = protocolVersion
        self.engineVersion = engineVersion
        self.device = device
        self.torchInstalled = torchInstalled
        self.modelsReady = modelsReady
        self.capabilities = capabilities
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        protocolVersion = try c.decode(Int.self, forKey: .protocolVersion)
        engineVersion = try c.decode(String.self, forKey: .engineVersion)
        device = try c.decodeIfPresent(String.self, forKey: .device)
        torchInstalled = try c.decode(Bool.self, forKey: .torchInstalled)
        modelsReady = try c.decode(Bool.self, forKey: .modelsReady)
        capabilities = try c.decodeIfPresent([String].self, forKey: .capabilities) ?? []
    }
}

// MARK: - Project

public struct Project: Codable, Equatable, Sendable {
    public var path: String
    public var name: String
    public var pages: [Page]

    public init(path: String, name: String, pages: [Page]) {
        self.path = path
        self.name = name
        self.pages = pages
    }
}

public struct Page: Codable, Equatable, Sendable, Identifiable {
    public var name: String
    public var source: String
    public var width: Int
    public var height: Int
    public var result: String?
    public var inpainted: String?
    public var done: Bool
    /// Increases whenever the result image is rewritten; use it to bust image caches.
    public var revision: Int
    public var blocks: [Block]

    public var id: String { name }
    public var size: CGSize { CGSize(width: width, height: height) }
    public var reviewCount: Int { blocks.count(where: \.needsReview) }

    public init(name: String, source: String, width: Int, height: Int, result: String? = nil,
                inpainted: String? = nil, done: Bool = false, revision: Int = 0, blocks: [Block] = []) {
        self.name = name
        self.source = source
        self.width = width
        self.height = height
        self.result = result
        self.inpainted = inpainted
        self.done = done
        self.revision = revision
        self.blocks = blocks
    }
}

public enum BlockAlignment: String, Codable, Sendable, CaseIterable {
    case left, center, right
}

/// `[r, g, b]`, 0–255.
public struct RGB: Codable, Equatable, Sendable {
    public var r: Int
    public var g: Int
    public var b: Int

    public init(r: Int, g: Int, b: Int) {
        self.r = r
        self.g = g
        self.b = b
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.singleValueContainer().decode([Double].self)
        guard values.count >= 3 else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "color needs 3 components"))
        }
        r = Int(values[0].rounded())
        g = Int(values[1].rounded())
        b = Int(values[2].rounded())
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode([r, g, b])
    }
}

/// `[x, y, w, h]` in source-image pixels.
private func decodeRect(_ values: [Double], codingPath: [CodingKey]) throws -> CGRect {
    guard values.count == 4 else {
        throw DecodingError.dataCorrupted(.init(codingPath: codingPath, debugDescription: "rect needs 4 numbers"))
    }
    return CGRect(x: values[0], y: values[1], width: values[2], height: values[3])
}

private func encodeRect(_ rect: CGRect) -> [Double] {
    [rect.origin.x, rect.origin.y, rect.width, rect.height]
}

public struct Block: Codable, Equatable, Sendable, Identifiable {
    public var index: Int
    /// Unrotated render box in image pixels.
    public var rect: CGRect
    public var sourceRect: CGRect?
    public var reviewReason: String?
    public var angle: Double
    public var sourceText: String
    public var translation: String
    public var vertical: Bool
    public var fontFamily: String
    /// Image pixels.
    public var fontSize: Double
    public var color: RGB
    public var alignment: BlockAlignment
    public var bold: Bool
    public var italic: Bool

    public var id: Int { index }
    /// The translation is not drawn; the original text is still on the page.
    public var isWithheld: Bool { reviewReason != nil }
    /// Withheld because of a problem, not on purpose and not as unreadable text outside a bubble.
    public var needsReview: Bool { isWithheld && reviewReason != Self.protected && !isSkipped }
    /// Unreadable text outside any bubble, left as the original.
    public var isSkipped: Bool { reviewReason == Self.skipped }
    /// Where the block shows on the page: the detected box while withheld, the render box otherwise.
    public var visibleRect: CGRect { isWithheld ? sourceRect ?? rect : rect }

    /// `review_reason` of a block that is left untranslated on purpose.
    public static let protected = "protected"
    /// `review_reason` of unreadable text outside any bubble (artwork, sound effects,
    /// signatures): left as the original and not something to review.
    public static let skipped = "skipped"

    enum CodingKeys: String, CodingKey {
        case index, rect, angle
        case sourceRect = "source_rect"
        case reviewReason = "review_reason"
        case sourceText = "source_text"
        case translation, vertical
        case fontFamily = "font_family"
        case fontSize = "font_size"
        case color, alignment, bold, italic
    }

    public init(index: Int, rect: CGRect, angle: Double = 0, sourceText: String = "", translation: String = "",
                vertical: Bool = false, fontFamily: String = "", fontSize: Double = 24,
                color: RGB = RGB(r: 0, g: 0, b: 0), alignment: BlockAlignment = .center,
                bold: Bool = false, italic: Bool = false, sourceRect: CGRect? = nil, reviewReason: String? = nil) {
        self.index = index
        self.rect = rect
        self.sourceRect = sourceRect
        self.reviewReason = reviewReason
        self.angle = angle
        self.sourceText = sourceText
        self.translation = translation
        self.vertical = vertical
        self.fontFamily = fontFamily
        self.fontSize = fontSize
        self.color = color
        self.alignment = alignment
        self.bold = bold
        self.italic = italic
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        index = try c.decode(Int.self, forKey: .index)
        rect = try decodeRect(try c.decode([Double].self, forKey: .rect), codingPath: c.codingPath + [CodingKeys.rect])
        sourceRect = try c.decodeIfPresent([Double].self, forKey: .sourceRect)
            .map { try decodeRect($0, codingPath: c.codingPath + [CodingKeys.sourceRect]) }
        reviewReason = try c.decodeIfPresent(String.self, forKey: .reviewReason)
        angle = try c.decode(Double.self, forKey: .angle)
        sourceText = try c.decode(String.self, forKey: .sourceText)
        translation = try c.decode(String.self, forKey: .translation)
        vertical = try c.decode(Bool.self, forKey: .vertical)
        fontFamily = try c.decode(String.self, forKey: .fontFamily)
        fontSize = try c.decode(Double.self, forKey: .fontSize)
        color = try c.decode(RGB.self, forKey: .color)
        alignment = try c.decode(BlockAlignment.self, forKey: .alignment)
        bold = try c.decode(Bool.self, forKey: .bold)
        italic = try c.decode(Bool.self, forKey: .italic)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(index, forKey: .index)
        try c.encode(encodeRect(rect), forKey: .rect)
        try c.encodeIfPresent(sourceRect.map(encodeRect), forKey: .sourceRect)
        try c.encodeIfPresent(reviewReason, forKey: .reviewReason)
        try c.encode(angle, forKey: .angle)
        try c.encode(sourceText, forKey: .sourceText)
        try c.encode(translation, forKey: .translation)
        try c.encode(vertical, forKey: .vertical)
        try c.encode(fontFamily, forKey: .fontFamily)
        try c.encode(fontSize, forKey: .fontSize)
        try c.encode(color, forKey: .color)
        try c.encode(alignment, forKey: .alignment)
        try c.encode(bold, forKey: .bold)
        try c.encode(italic, forKey: .italic)
    }
}

/// One entry of `update_blocks.edits`; nil fields are left untouched.
public struct BlockEdit: Codable, Equatable, Sendable {
    public var index: Int
    public var sourceText: String?
    public var translation: String?
    public var rect: CGRect?
    public var angle: Double?
    public var fontFamily: String?
    public var fontSize: Double?
    public var color: RGB?
    public var alignment: BlockAlignment?
    public var bold: Bool?
    public var italic: Bool?
    /// `true` leaves the block untranslated on purpose; `false` withdraws that.
    public var keepOriginal: Bool?

    enum CodingKeys: String, CodingKey {
        case index, translation, rect, angle
        case sourceText = "source_text"
        case fontFamily = "font_family"
        case fontSize = "font_size"
        case color, alignment, bold, italic
        case keepOriginal = "keep_original"
    }

    public init(index: Int, translation: String? = nil, rect: CGRect? = nil, angle: Double? = nil,
                fontFamily: String? = nil, fontSize: Double? = nil, color: RGB? = nil,
                alignment: BlockAlignment? = nil, bold: Bool? = nil, italic: Bool? = nil,
                sourceText: String? = nil, keepOriginal: Bool? = nil) {
        self.index = index
        self.keepOriginal = keepOriginal
        self.sourceText = sourceText
        self.translation = translation
        self.rect = rect
        self.angle = angle
        self.fontFamily = fontFamily
        self.fontSize = fontSize
        self.color = color
        self.alignment = alignment
        self.bold = bold
        self.italic = italic
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        index = try c.decode(Int.self, forKey: .index)
        sourceText = try c.decodeIfPresent(String.self, forKey: .sourceText)
        translation = try c.decodeIfPresent(String.self, forKey: .translation)
        rect = try c.decodeIfPresent([Double].self, forKey: .rect)
            .map { try decodeRect($0, codingPath: c.codingPath + [CodingKeys.rect]) }
        angle = try c.decodeIfPresent(Double.self, forKey: .angle)
        fontFamily = try c.decodeIfPresent(String.self, forKey: .fontFamily)
        fontSize = try c.decodeIfPresent(Double.self, forKey: .fontSize)
        color = try c.decodeIfPresent(RGB.self, forKey: .color)
        alignment = try c.decodeIfPresent(BlockAlignment.self, forKey: .alignment)
        bold = try c.decodeIfPresent(Bool.self, forKey: .bold)
        italic = try c.decodeIfPresent(Bool.self, forKey: .italic)
        keepOriginal = try c.decodeIfPresent(Bool.self, forKey: .keepOriginal)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(index, forKey: .index)
        try c.encodeIfPresent(sourceText, forKey: .sourceText)
        try c.encodeIfPresent(translation, forKey: .translation)
        try c.encodeIfPresent(rect.map(encodeRect), forKey: .rect)
        try c.encodeIfPresent(angle, forKey: .angle)
        try c.encodeIfPresent(fontFamily, forKey: .fontFamily)
        try c.encodeIfPresent(fontSize, forKey: .fontSize)
        try c.encodeIfPresent(color, forKey: .color)
        try c.encodeIfPresent(alignment, forKey: .alignment)
        try c.encodeIfPresent(bold, forKey: .bold)
        try c.encodeIfPresent(italic, forKey: .italic)
        try c.encodeIfPresent(keepOriginal, forKey: .keepOriginal)
    }

    /// Later values win; used to coalesce queued edits of one block.
    public func merging(_ other: BlockEdit) -> BlockEdit {
        var merged = self
        merged.sourceText = other.sourceText ?? sourceText
        merged.translation = other.translation ?? translation
        merged.rect = other.rect ?? rect
        merged.angle = other.angle ?? angle
        merged.fontFamily = other.fontFamily ?? fontFamily
        merged.fontSize = other.fontSize ?? fontSize
        merged.color = other.color ?? color
        merged.alignment = other.alignment ?? alignment
        merged.bold = other.bold ?? bold
        merged.italic = other.italic ?? italic
        merged.keepOriginal = other.keepOriginal ?? keepOriginal
        return merged
    }

    public func apply(to block: inout Block) {
        if let sourceText { block.sourceText = sourceText }
        if let translation { block.translation = translation }
        if let rect { block.rect = rect }
        if let angle { block.angle = angle }
        if let fontFamily { block.fontFamily = fontFamily }
        if let fontSize { block.fontSize = fontSize }
        if let color { block.color = color }
        if let alignment { block.alignment = alignment }
        if let bold { block.bold = bold }
        if let italic { block.italic = italic }
        if keepOriginal == true {
            block.reviewReason = Block.protected
        } else if keepOriginal == false, block.reviewReason == Block.protected {
            // Withheld again; the engine's reply carries the actual reason.
            block.reviewReason = "untranslated"
        }
    }

    /// The edit that restores `block`'s current values for every field this edit sets.
    public func inverse(from block: Block) -> BlockEdit {
        BlockEdit(
            index: index,
            translation: translation.map { _ in block.translation },
            rect: rect.map { _ in block.rect },
            angle: angle.map { _ in block.angle },
            fontFamily: fontFamily.map { _ in block.fontFamily },
            fontSize: fontSize.map { _ in block.fontSize },
            color: color.map { _ in block.color },
            alignment: alignment.map { _ in block.alignment },
            bold: bold.map { _ in block.bold },
            italic: italic.map { _ in block.italic },
            sourceText: sourceText.map { _ in block.sourceText },
            keepOriginal: keepOriginal.map { !$0 }
        )
    }
}

// MARK: - Fonts

public struct FontInfo: Codable, Equatable, Sendable, Identifiable {
    public var family: String
    public var display: String
    /// `"system"` or `"custom"`.
    public var source: String

    public var id: String { family }

    public init(family: String, display: String, source: String) {
        self.family = family
        self.display = display
        self.source = source
    }
}

public struct FontList: Codable, Equatable, Sendable {
    public var fonts: [FontInfo]
    public var `default`: String

    public init(fonts: [FontInfo], default: String) {
        self.fonts = fonts
        self.default = `default`
    }
}

// MARK: - Settings

public enum TranslatorKind: String, Codable, Sendable {
    case google, llm, other

    public init(from decoder: Decoder) throws {
        self = TranslatorKind(rawValue: try decoder.singleValueContainer().decode(String.self)) ?? .other
    }
}

public struct TranslatorSelection: Codable, Equatable, Sendable {
    public var kind: TranslatorKind
    public var module: String
    public var profileID: String?

    enum CodingKeys: String, CodingKey {
        case kind, module
        case profileID = "profile_id"
    }

    public init(kind: TranslatorKind, module: String, profileID: String? = nil) {
        self.kind = kind
        self.module = module
        self.profileID = profileID
    }
}

/// `Profile.vision`: whether the service was measured to read images.
public enum VisionSupport: String, Codable, Sendable {
    case yes, no, unknown

    public init(from decoder: Decoder) throws {
        self = VisionSupport(rawValue: try decoder.singleValueContainer().decode(String.self)) ?? .unknown
    }
}

public struct Profile: Codable, Equatable, Sendable, Identifiable {
    public var id: String
    public var name: String
    public var baseURL: String
    public var model: String
    public var modelOptions: [String]
    public var requireAPIKey: Bool
    /// The key itself never leaves the engine.
    public var hasAPIKey: Bool
    /// Measured by `probe_vision`; `.unknown` when unmeasured or sent by an older engine.
    public var vision: VisionSupport

    enum CodingKeys: String, CodingKey {
        case id, name
        case baseURL = "base_url"
        case model
        case modelOptions = "model_options"
        case requireAPIKey = "require_api_key"
        case hasAPIKey = "has_api_key"
        case vision
    }

    public init(id: String, name: String, baseURL: String, model: String, modelOptions: [String] = [],
                requireAPIKey: Bool = true, hasAPIKey: Bool = false, vision: VisionSupport = .unknown) {
        self.id = id
        self.name = name
        self.baseURL = baseURL
        self.model = model
        self.modelOptions = modelOptions
        self.requireAPIKey = requireAPIKey
        self.hasAPIKey = hasAPIKey
        self.vision = vision
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(String.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        baseURL = try c.decode(String.self, forKey: .baseURL)
        model = try c.decode(String.self, forKey: .model)
        modelOptions = try c.decode([String].self, forKey: .modelOptions)
        requireAPIKey = try c.decode(Bool.self, forKey: .requireAPIKey)
        hasAPIKey = try c.decode(Bool.self, forKey: .hasAPIKey)
        vision = try c.decodeIfPresent(VisionSupport.self, forKey: .vision) ?? .unknown
    }
}

/// `get_settings.enhance`: re-check hard blocks with a vision model after a page is composed.
public struct EnhanceSettings: Codable, Equatable, Sendable {
    public var enabled: Bool
    /// nil uses the translation service's profile.
    public var profileID: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case profileID = "profile_id"
    }

    public init(enabled: Bool = false, profileID: String? = nil) {
        self.enabled = enabled
        self.profileID = profileID
    }
}

public struct LanguageOptions: Codable, Equatable, Sendable {
    public var source: [String]
    public var target: [String]

    public init(source: [String], target: [String]) {
        self.source = source
        self.target = target
    }
}

public struct OutputSettings: Codable, Equatable, Sendable {
    /// `.png`, `.jpg` or `.webp`.
    public var format: String
    public var quality: Int

    public init(format: String, quality: Int) {
        self.format = format
        self.quality = quality
    }
}

public struct EngineSettings: Codable, Equatable, Sendable {
    public var sourceLanguage: String
    public var targetLanguage: String
    public var languages: LanguageOptions
    public var translator: TranslatorSelection
    public var profiles: [Profile]
    public var providerTemplates: [String]
    public var output: OutputSettings
    /// Off when the engine predates the `enhance` capability.
    public var enhance: EnhanceSettings

    enum CodingKeys: String, CodingKey {
        case sourceLanguage = "source_language"
        case targetLanguage = "target_language"
        case languages, translator, profiles
        case providerTemplates = "provider_templates"
        case output, enhance
    }

    public init(sourceLanguage: String, targetLanguage: String, languages: LanguageOptions,
                translator: TranslatorSelection, profiles: [Profile], providerTemplates: [String],
                output: OutputSettings, enhance: EnhanceSettings = EnhanceSettings()) {
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        self.languages = languages
        self.translator = translator
        self.profiles = profiles
        self.providerTemplates = providerTemplates
        self.output = output
        self.enhance = enhance
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        sourceLanguage = try c.decode(String.self, forKey: .sourceLanguage)
        targetLanguage = try c.decode(String.self, forKey: .targetLanguage)
        languages = try c.decode(LanguageOptions.self, forKey: .languages)
        translator = try c.decode(TranslatorSelection.self, forKey: .translator)
        profiles = try c.decode([Profile].self, forKey: .profiles)
        providerTemplates = try c.decode([String].self, forKey: .providerTemplates)
        output = try c.decode(OutputSettings.self, forKey: .output)
        enhance = try c.decodeIfPresent(EnhanceSettings.self, forKey: .enhance) ?? EnhanceSettings()
    }

    /// The profile that re-checks pages: the chosen one, else the translation service's.
    /// nil when neither names an existing profile (e.g. Google translates and none is chosen).
    public var enhanceProfile: Profile? {
        let id = enhance.profileID ?? (translator.kind == .llm ? translator.profileID : nil)
        return profiles.first { $0.id == id }
    }

    /// Whether enhanced mode can be on with the current choices.
    public var enhanceAvailability: EnhanceAvailability {
        guard let profile = enhanceProfile else { return .noProfile }
        switch profile.vision {
        case .yes: return .available
        case .no: return .visionUnsupported
        case .unknown: return .visionUnknown(profileID: profile.id)
        }
    }

    /// What the toggle shows: a stored "on" whose service can no longer read images shows as off.
    public var enhanceIsOn: Bool {
        enhance.enabled && enhanceAvailability == .available
    }
}

public enum EnhanceAvailability: Equatable, Sendable {
    case available
    /// The effective profile was measured not to read images.
    case visionUnsupported
    /// The effective profile has not been measured yet; `probe_vision` can.
    case visionUnknown(profileID: String)
    /// No LLM profile to re-check with.
    case noProfile
}

/// Value for `set_settings.translator`.
public enum TranslatorChoice: Encodable, Equatable, Sendable {
    case google
    case llm(profileID: String)

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: TranslatorSelection.CodingKeys.self)
        switch self {
        case .google:
            try c.encode("google", forKey: .kind)
        case .llm(let profileID):
            try c.encode("llm", forKey: .kind)
            try c.encode(profileID, forKey: .profileID)
        }
    }
}

/// Params of `set_settings`; nil fields are left untouched.
public struct SettingsPatch: Encodable, Equatable, Sendable {
    public var sourceLanguage: String?
    public var targetLanguage: String?
    public var translator: TranslatorChoice?
    public var output: OutputSettings?
    public var enhance: EnhancePatch?

    enum CodingKeys: String, CodingKey {
        case sourceLanguage = "source_language"
        case targetLanguage = "target_language"
        case translator, output, enhance
    }

    public init(sourceLanguage: String? = nil, targetLanguage: String? = nil,
                translator: TranslatorChoice? = nil, output: OutputSettings? = nil, enhance: EnhancePatch? = nil) {
        self.sourceLanguage = sourceLanguage
        self.targetLanguage = targetLanguage
        self.translator = translator
        self.output = output
        self.enhance = enhance
    }
}

/// Value for `set_settings.enhance`; nil fields are left untouched.
public struct EnhancePatch: Encodable, Equatable, Sendable {
    public var enabled: Bool?
    /// `.some(nil)` sends `null`: use the translation service's profile.
    public var profileID: String??

    public init(enabled: Bool? = nil, profileID: String?? = nil) {
        self.enabled = enabled
        self.profileID = profileID
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: EnhanceSettings.CodingKeys.self)
        try c.encodeIfPresent(enabled, forKey: .enabled)
        if let profileID {
            if let profileID { try c.encode(profileID, forKey: .profileID) } else { try c.encodeNil(forKey: .profileID) }
        }
    }
}

/// Params of `update_profile` besides `id`. `apiKey` is write-only; `""` clears it.
public struct ProfilePatch: Encodable, Equatable, Sendable {
    public var name: String?
    public var baseURL: String?
    public var model: String?
    public var apiKey: String?

    enum CodingKeys: String, CodingKey {
        case name
        case baseURL = "base_url"
        case model
        case apiKey = "api_key"
    }

    public init(name: String? = nil, baseURL: String? = nil, model: String? = nil, apiKey: String? = nil) {
        self.name = name
        self.baseURL = baseURL
        self.model = model
        self.apiKey = apiKey
    }

    public var isEmpty: Bool { self == ProfilePatch() }
}

public struct ProfileTestResult: Codable, Equatable, Sendable {
    public var ok: Bool
    public var message: String
    public var kind: String
    public var models: [String]

    public init(ok: Bool, message: String, kind: String, models: [String]) {
        self.ok = ok
        self.message = message
        self.kind = kind
        self.models = models
    }
}

/// `probe_vision` reply: a measurement, or the `test_profile`-shaped failure when the
/// service could not be reached.
public enum VisionProbeResult: Decodable, Equatable, Sendable {
    case measured(VisionSupport)
    case failed(ProfileTestResult)

    enum CodingKeys: String, CodingKey {
        case vision
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let vision = try c.decodeIfPresent(VisionSupport.self, forKey: .vision) {
            self = .measured(vision)
        } else {
            self = .failed(try ProfileTestResult(from: decoder))
        }
    }
}

public enum RunMode: String, Codable, Sendable {
    case all
    case `continue`
}

public enum ExportKind: String, Codable, Sendable {
    case images
    case translationText = "translation_txt"
    case sourceText = "source_txt"
}

public struct ExportResult: Codable, Equatable, Sendable {
    public var files: [String]
}

// MARK: - Events

public struct PrepareProgress: Codable, Equatable, Sendable {
    /// `detector`, `ocr`, `inpainter`, `translator` or `packages`.
    public var module: String
    /// e.g. `downloading`, `installing`, `loading`.
    public var phase: String
    public var message: String
    /// Bytes, when known.
    public var done: Double?
    public var total: Double?

    public init(module: String, phase: String, message: String, done: Double? = nil, total: Double? = nil) {
        self.module = module
        self.phase = phase
        self.message = message
        self.done = done
        self.total = total
    }
}

public enum EngineEvent: Equatable, Sendable {
    case prepareProgress(PrepareProgress)
    /// `stage` is `detect`, `ocr`, `translate`, `inpaint` or `enhance`; counts are pages.
    case stageProgress(stage: String, done: Int, total: Int)
    case pageDone(page: String, revision: Int)
    case runFinished(cancelled: Bool, failedPages: [String])
    case error(message: String, page: String?)
    case log(level: String, message: String)
    /// Not part of the wire protocol: the engine process ended without being asked to.
    case terminated(status: Int32?)
}

extension EngineEvent: Decodable {
    enum CodingKeys: String, CodingKey {
        case event, stage, done, total, page, revision, cancelled
        case failedPages = "failed_pages"
        case message, level
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let name = try c.decode(String.self, forKey: .event)
        switch name {
        case "prepare_progress":
            self = .prepareProgress(try PrepareProgress(from: decoder))
        case "stage_progress":
            self = .stageProgress(stage: try c.decode(String.self, forKey: .stage),
                                  done: try c.decode(Int.self, forKey: .done),
                                  total: try c.decode(Int.self, forKey: .total))
        case "page_done":
            self = .pageDone(page: try c.decode(String.self, forKey: .page),
                             revision: try c.decode(Int.self, forKey: .revision))
        case "run_finished":
            self = .runFinished(cancelled: try c.decode(Bool.self, forKey: .cancelled),
                                failedPages: try c.decodeIfPresent([String].self, forKey: .failedPages) ?? [])
        case "error":
            self = .error(message: try c.decode(String.self, forKey: .message),
                          page: try c.decodeIfPresent(String.self, forKey: .page))
        case "log":
            self = .log(level: try c.decodeIfPresent(String.self, forKey: .level) ?? "info",
                        message: try c.decode(String.self, forKey: .message))
        default:
            throw DecodingError.dataCorruptedError(forKey: .event, in: c, debugDescription: "Unknown event \(name)")
        }
    }
}

// MARK: - Errors

/// Stable `error.code` strings.
public enum ErrorCode: String, Sendable, CaseIterable {
    case badRequest = "bad_request"
    case notFound = "not_found"
    case busy
    case noProject = "no_project"
    case translatorNotReady = "translator_not_ready"
    case `internal`
}

public enum EngineError: Error, Equatable, Sendable {
    /// An `error` response from the engine. `message` is safe to show.
    case remote(code: String, message: String, data: JSONValue?)
    case notRunning
    /// The process ended while the request was pending.
    case terminated(status: Int32?)
    case launchFailed(String)
    case invalidResponse(String)

    public var code: ErrorCode? {
        if case .remote(let code, _, _) = self { ErrorCode(rawValue: code) } else { nil }
    }

    /// `error.data.missing` of `translator_not_ready`: `api_key`, `model` or `base_url`.
    public var missingTranslatorField: String? {
        if case .remote(_, _, let data) = self { data?["missing"]?.stringValue } else { nil }
    }
}

extension EngineError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .remote(_, let message, _): message
        case .notRunning: String(localized: "The engine is not running.")
        case .terminated: String(localized: "The engine quit unexpectedly.")
        case .launchFailed(let reason): reason
        case .invalidResponse: String(localized: "The engine sent an unexpected reply.")
        }
    }
}
