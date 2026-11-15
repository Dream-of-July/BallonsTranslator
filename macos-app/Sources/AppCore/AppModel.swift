import EngineKit
import Foundation
import Observation

/// App state for the single window: engine status, the open project, the running
/// translation and block edits. Views read it; every engine call goes through here.
@MainActor
@Observable
public final class AppModel {
    public enum EngineState: Equatable {
        case starting
        case ready(Hello)
        case failed(String)
    }

    public struct RunState: Equatable {
        public var pages: Set<String>
        /// `detect`, `ocr`, `translate`, `inpaint`, `enhance`; nil until the first stage reports.
        public var stage: String?
        public var done = 0
        public var total = 0
        /// First-run downloads and model loading.
        public var prepare: PrepareProgress?
        public var cancelling = false
    }

    public enum PageStatus: Sendable {
        case pending, inProgress, done, failed
    }

    /// An inline, dismissible message with at most one way to fix it.
    public struct Notice: Equatable {
        public enum Action: Equatable {
            case none
            case openTranslationSettings
        }

        public var message: String
        public var action = Action.none
    }

    public enum ProfileTest: Equatable {
        case running
        case finished(ProfileTestResult)
    }

    /// A `probe_vision` in flight, or the reason the last one gave no measurement.
    /// A completed measurement lands in the profile's `vision` instead.
    public enum VisionProbe: Equatable {
        case running
        case failed(String)
    }

    public private(set) var engineState = EngineState.starting
    public private(set) var project: Project?
    public private(set) var isOpening = false
    public private(set) var settings: EngineSettings?
    public private(set) var fonts: FontList?
    public private(set) var run: RunState?
    public private(set) var failedPages: Set<String> = []
    private var finishedInRun: Set<String> = []
    /// Pages whose result image is being re-rendered after an edit.
    public private(set) var renderingPages: Set<String> = []
    public private(set) var profileTests: [String: ProfileTest] = [:]
    public private(set) var visionProbes: [String: VisionProbe] = [:]

    public var selectedPage: String? {
        didSet { if selectedPage != oldValue { selectedBlock = nil } }
    }
    public var selectedBlock: Int?
    public var notice: Notice?
    public var showsOriginal = false
    /// Changes whenever the selected block should be scrolled into view.
    public private(set) var revealRequest = 0
    /// Multiplier on the fit-to-window scale; 1 shows the whole page.
    public var zoom = 1.0
    /// Profile to show when Settings opens after a "translator not ready" error.
    public var settingsProfile: String?
    /// The window's undo manager; set by the main view.
    public weak var undoManager: UndoManager?

    @ObservationIgnored private let engine: any Engine
    @ObservationIgnored private let defaults: UserDefaults?
    @ObservationIgnored private var eventTask: Task<Void, Never>?
    @ObservationIgnored private var queue = EditQueue()
    @ObservationIgnored private var deletingPages: Set<String> = []
    @ObservationIgnored private var pendingOpen: [URL]?

    private static let lastProjectKey = "lastProjectPath"

    /// - Parameter defaults: where the last project is remembered; nil remembers nothing.
    public init(engine: any Engine, defaults: UserDefaults? = nil) {
        self.engine = engine
        self.defaults = defaults
    }

    // MARK: - Engine lifecycle

    public var isEngineReady: Bool {
        if case .ready = engineState { true } else { false }
    }

    /// Starts the engine and loads settings and fonts. Returns once it is ready or has failed.
    public func start() async {
        if eventTask == nil {
            let events = engine.events
            eventTask = Task { [weak self] in
                for await event in events {
                    self?.handle(event)
                }
            }
        }
        engineState = .starting
        do {
            try await engine.start()
            let hello = try await engine.hello()
            guard hello.protocolVersion == engineProtocolVersion else {
                engineState = .failed(String(localized: "The engine and the app are different versions."))
                return
            }
            settings = try await engine.settings()
            fonts = try await engine.listFonts()
            engineState = .ready(hello)
        } catch {
            engineState = .failed(error.localizedDescription)
            return
        }
        if let urls = pendingOpen {
            pendingOpen = nil
            await open(urls)
        } else if let path = project?.path ?? defaults?.string(forKey: Self.lastProjectKey),
                  FileManager.default.fileExists(atPath: path) {
            await open([URL(fileURLWithPath: path)])
        }
    }

    public func restartEngine() async {
        await engine.stop()
        resetTransientState()
        await start()
    }

    public func logLines() async -> [String] {
        await engine.diagnostics()
    }

    private func resetTransientState() {
        run = nil
        queue.reset()
        renderingPages = []
        deletingPages = []
    }

    private func handle(_ event: EngineEvent) {
        switch event {
        case .prepareProgress(let progress):
            run?.prepare = progress
        case .stageProgress(let stage, let done, let total):
            run?.prepare = nil
            run?.stage = stage
            run?.done = done
            run?.total = total
        case .pageDone(let page, _):
            failedPages.remove(page)
            finishedInRun.insert(page)
            Task { await refresh(page) }
        case .runFinished(_, let failed):
            let pages = run?.pages ?? []
            run = nil
            failedPages.formUnion(failed)
            if !failed.isEmpty, notice == nil {
                notice = Notice(message: failed.count == 1
                    ? String(localized: "1 page couldn't be translated.")
                    : String(localized: "\(failed.count) pages couldn't be translated."))
            }
            Task {
                for page in pages { await refresh(page) }
            }
        case .error(let message, let page):
            if let page { failedPages.insert(page) }
            notice = Notice(message: message)
        case .log:
            break
        case .terminated:
            resetTransientState()
            engineState = .failed(String(localized: "The engine quit unexpectedly."))
        }
    }

    // MARK: - Project

    public var currentPage: Page? {
        project?.pages.first { $0.name == selectedPage }
    }

    public var currentBlock: Block? {
        currentPage?.blocks.first { $0.index == selectedBlock }
    }

    /// Opens a dropped or chosen selection: a folder in place, or loose images copied into a new project.
    public func open(_ urls: [URL]) async {
        guard isEngineReady else {
            pendingOpen = urls
            return
        }
        guard run == nil, !isOpening else { return }
        isOpening = true
        defer { isOpening = false }
        do {
            let folder: URL
            if let directory = urls.first(where: ProjectImporter.isDirectory) {
                folder = directory
            } else if urls.contains(where: ProjectImporter.isImage) {
                folder = try await ProjectImporter.importImages(urls)
            } else {
                notice = Notice(message: String(localized: "Drop a folder or image files."))
                return
            }
            let opened = try await engine.openProject(path: folder.path)
            undoManager?.removeAllActions(withTarget: self)
            queue.reset()
            renderingPages = []
            failedPages = []
            notice = nil
            project = opened
            selectedPage = opened.pages.first?.name
            showsOriginal = false
            zoom = 1
            defaults?.set(opened.path, forKey: Self.lastProjectKey)
        } catch {
            present(error)
        }
    }

    public func status(of page: Page) -> PageStatus {
        if failedPages.contains(page.name) { return .failed }
        // A page of the current run counts as in progress until its result arrives.
        if let run, run.pages.contains(page.name), !finishedInRun.contains(page.name) { return .inProgress }
        return page.done ? .done : .pending
    }

    private func refresh(_ name: String) async {
        guard let page = try? await engine.page(named: name) else { return }
        replace(page)
    }

    private func replace(_ page: Page) {
        guard let position = project?.pages.firstIndex(where: { $0.name == page.name }) else { return }
        var page = page
        // Edits queued behind the request that produced `page` are not in it yet.
        for edit in queue.waitingEdits(page: page.name) {
            if let block = page.blocks.firstIndex(where: { $0.index == edit.index }) {
                edit.apply(to: &page.blocks[block])
            }
        }
        project?.pages[position] = page
        if page.name == selectedPage, let selectedBlock, !page.blocks.contains(where: { $0.index == selectedBlock }) {
            self.selectedBlock = nil
        }
    }

    // MARK: - Review

    /// Blocks of a translated page whose translation was withheld and still needs a look.
    public func reviewCount(of page: Page) -> Int {
        page.result == nil ? 0 : page.reviewCount
    }

    public var translatedPageCount: Int {
        project?.pages.count(where: \.done) ?? 0
    }

    public var reviewTotal: Int {
        project?.pages.reduce(0) { $0 + reviewCount(of: $1) } ?? 0
    }

    /// Selects the next block to review on the current page, wrapping around; when the page
    /// has none left, the first one on the next page that has some. Returns false when nothing is left.
    @discardableResult
    public func selectNextReview() -> Bool {
        guard let pages = project?.pages,
              let current = pages.firstIndex(where: { $0.name == selectedPage }) else { return false }
        if reviewCount(of: pages[current]) > 0 {
            let waiting = pages[current].blocks.filter(\.needsReview).map(\.index)
            let next = waiting.firstIndex { $0 == selectedBlock }.map { ($0 + 1) % waiting.count } ?? 0
            selectedBlock = waiting[next]
        } else {
            let following = (1..<pages.count).map { pages[(current + $0) % pages.count] }
            guard let page = following.first(where: { reviewCount(of: $0) > 0 }),
                  let block = page.blocks.first(where: \.needsReview) else { return false }
            selectedPage = page.name
            selectedBlock = block.index
        }
        showsOriginal = false
        revealRequest += 1
        return true
    }

    // MARK: - Translate / export

    public var canTranslate: Bool {
        isEngineReady && run == nil && !(project?.pages.isEmpty ?? true)
    }

    /// Translates the given pages, or the whole project: only the remaining pages when
    /// some are already done, everything otherwise.
    public func translate(pages: [String]? = nil) async {
        guard canTranslate, let project else { return }
        let done = Set(project.pages.filter(\.done).map(\.name))
        let all = project.pages.map(\.name)
        let mode: RunMode = pages == nil && !done.isEmpty && done.count < all.count ? .continue : .all
        var targets = Set(pages ?? all)
        if mode == .continue { targets.subtract(done) }
        notice = nil
        selectedBlock = nil
        failedPages.subtract(targets)
        finishedInRun = []
        run = RunState(pages: targets)
        do {
            try await engine.run(pages: pages, mode: mode)
        } catch {
            run = nil
            present(error)
        }
    }

    public func cancelRun() async {
        guard run != nil, run?.cancelling == false else { return }
        run?.cancelling = true
        do {
            try await engine.cancel()
        } catch {
            run?.cancelling = false
            present(error)
        }
    }

    /// Copies the result images into `folder`. Returns the written files, or nil on failure.
    public func export(to folder: URL) async -> [String]? {
        do {
            let files = try await engine.export(to: folder.path, what: .images)
            if files.isEmpty {
                notice = Notice(message: String(localized: "Nothing to export yet."))
            }
            return files
        } catch {
            present(error)
            return nil
        }
    }

    private func present(_ error: Error) {
        if let error = error as? EngineError, error.code == .translatorNotReady {
            settingsProfile = settings?.translator.profileID
            notice = Notice(message: error.localizedDescription, action: .openTranslationSettings)
        } else if !(error is CancellationError) {
            notice = Notice(message: error.localizedDescription)
        }
    }

    // MARK: - Block edits

    public func canEdit(page: String) -> Bool {
        isEngineReady && run == nil && !deletingPages.contains(page)
    }

    /// Applies `edit` to the local page at once, registers undo, and queues the engine request.
    /// With `autofit` the engine also re-lays-out the block (its box and font size may change).
    public func apply(_ edit: BlockEdit, autofit: Bool = false, to pageName: String) {
        guard canEdit(page: pageName),
              let page = project?.pages.firstIndex(where: { $0.name == pageName }),
              let position = project?.pages[page].blocks.firstIndex(where: { $0.index == edit.index }),
              let block = project?.pages[page].blocks[position] else { return }
        var edited = block
        edit.apply(to: &edited)
        guard edited != block || autofit else { return }

        var inverse = edit.inverse(from: block)
        if autofit {
            inverse.rect = block.rect
            inverse.fontSize = block.fontSize
        }
        undoManager?.registerUndo(withTarget: self) { model in
            model.apply(inverse, to: pageName)
        }
        project?.pages[page].blocks[position] = edited
        if let batch = queue.enqueue(edit, autofit: autofit, page: pageName) {
            send(batch)
        }
    }

    /// Moves or resizes a block's box. A withheld block has nothing drawn, so it stays where it is.
    public func setRect(_ rect: CGRect, ofBlock index: Int, on pageName: String) {
        guard let block = project?.pages.first(where: { $0.name == pageName })?.blocks.first(where: { $0.index == index }),
              !block.isWithheld else { return }
        apply(BlockEdit(index: index, rect: rect), to: pageName)
    }

    /// Whether the engine can leave a block untranslated on purpose.
    public var canKeepOriginal: Bool { engineSupports(Hello.keepOriginal) }

    private func engineSupports(_ capability: String) -> Bool {
        if case .ready(let hello) = engineState { hello.capabilities.contains(capability) } else { false }
    }

    /// Leaves a withheld block as it is on the page; it no longer counts as "to review".
    public func keepOriginal(_ index: Int, on pageName: String) {
        guard canKeepOriginal else { return }
        apply(BlockEdit(index: index, keepOriginal: true), to: pageName)
    }

    /// Removes a block. Not undoable: protocol v1 has no way to bring a block back, and
    /// the remaining indexes may shift, so earlier undo steps are dropped as well.
    public func deleteBlock(_ index: Int, on pageName: String) {
        guard canEdit(page: pageName),
              let page = project?.pages.firstIndex(where: { $0.name == pageName }),
              project?.pages[page].blocks.contains(where: { $0.index == index }) == true else { return }
        project?.pages[page].blocks.removeAll { $0.index == index }
        if selectedPage == pageName, selectedBlock == index { selectedBlock = nil }
        undoManager?.removeAllActions(withTarget: self)
        deletingPages.insert(pageName)
        if let batch = queue.enqueueDeletion(of: index, page: pageName) {
            send(batch)
        }
    }

    private func send(_ first: EditQueue.Batch) {
        renderingPages.insert(first.page)
        Task {
            var next: EditQueue.Batch? = first
            while let batch = next {
                do {
                    if !batch.edits.isEmpty {
                        replace(try await engine.updateBlocks(page: batch.page, edits: batch.edits, autofit: batch.autofit))
                    }
                    if !batch.deletions.isEmpty {
                        replace(try await engine.deleteBlocks(page: batch.page, indexes: batch.deletions))
                    }
                } catch {
                    present(error)
                    // The optimistic local state is wrong now; take the engine's.
                    if let page = try? await engine.page(named: batch.page) { replace(page) }
                }
                next = queue.finish(page: batch.page)
            }
            renderingPages.remove(first.page)
            deletingPages.remove(first.page)
        }
    }

    // MARK: - Settings

    public func updateSettings(_ patch: SettingsPatch) async {
        do {
            settings = try await engine.setSettings(patch)
        } catch {
            present(error)
        }
    }

    public func createProfile(template: String) async -> Profile? {
        do {
            let profile = try await engine.createProfile(template: template)
            settings = try await engine.settings()
            return profile
        } catch {
            present(error)
            return nil
        }
    }

    public func updateProfile(id: String, _ patch: ProfilePatch) async {
        guard !patch.isEmpty else { return }
        do {
            let profile = try await engine.updateProfile(id: id, patch)
            if let position = settings?.profiles.firstIndex(where: { $0.id == id }) {
                settings?.profiles[position] = profile
            }
            profileTests[id] = nil
            visionProbes[id] = nil
        } catch {
            present(error)
        }
    }

    public func deleteProfile(id: String) async {
        do {
            settings = try await engine.deleteProfile(id: id)
            profileTests[id] = nil
            visionProbes[id] = nil
        } catch {
            present(error)
        }
    }

    public func testProfile(id: String) async {
        profileTests[id] = .running
        do {
            let result = try await engine.testProfile(id: id)
            profileTests[id] = .finished(result)
            // A successful test may have refreshed the profile's model list.
            if let refreshed = try? await engine.settings() { settings = refreshed }
        } catch {
            profileTests[id] = .finished(ProfileTestResult(ok: false, message: error.localizedDescription, kind: "error", models: []))
        }
    }

    /// Whether the engine has the enhanced-mode setting.
    public var supportsEnhance: Bool { engineSupports(Hello.enhance) }

    /// Whether the engine can measure a profile's image support.
    public var canProbeVision: Bool { engineSupports(Hello.visionProbe) }

    /// Measures whether a profile reads images. The result lands in the profile's `vision`;
    /// a service that could not be reached leaves it unchanged and keeps the reason in `visionProbes`.
    public func probeVision(profileID: String) async {
        guard canProbeVision, visionProbes[profileID] != .running else { return }
        visionProbes[profileID] = .running
        do {
            switch try await engine.probeVision(profileID: profileID) {
            case .measured(let vision):
                visionProbes[profileID] = nil
                if let position = settings?.profiles.firstIndex(where: { $0.id == profileID }) {
                    settings?.profiles[position].vision = vision
                }
            case .failed(let result):
                visionProbes[profileID] = .failed(result.message)
            }
        } catch {
            visionProbes[profileID] = .failed(error.localizedDescription)
        }
    }

    /// Turns enhanced mode on or off. Turning it on is refused unless the service can read images.
    public func setEnhanceEnabled(_ enabled: Bool) async {
        guard supportsEnhance, let settings, !enabled || settings.enhanceAvailability == .available else { return }
        await updateSettings(SettingsPatch(enhance: EnhancePatch(enabled: enabled)))
    }

    /// Chooses the profile that re-checks pages; nil uses the translation service's.
    /// A profile measured not to read images cannot be chosen.
    public func setEnhanceProfile(_ profileID: String?) async {
        guard supportsEnhance,
              profileID.flatMap({ id in settings?.profiles.first { $0.id == id } })?.vision != .no else { return }
        await updateSettings(SettingsPatch(enhance: EnhancePatch(profileID: .some(profileID))))
    }

    // MARK: - Zoom

    public static let zoomRange = 0.5...8.0

    public func zoomIn() { zoom = min(zoom * 1.25, Self.zoomRange.upperBound) }
    public func zoomOut() { zoom = max(zoom / 1.25, Self.zoomRange.lowerBound) }
    public func zoomToFit() { zoom = 1 }
}
