import CoreGraphics
import Foundation
import Testing
@testable import AppCore
@testable import EngineKit

private func isClose(_ a: CGPoint, _ b: CGPoint, tolerance: CGFloat = 1e-6) -> Bool {
    abs(a.x - b.x) < tolerance && abs(a.y - b.y) < tolerance
}

@Suite struct GeometryTests {
    @Test func viewAndImageSpaceRoundTripAtAnyZoom() {
        let image = CGSize(width: 800, height: 1200)
        let fit = PageGeometry.fitScale(image: image, container: CGSize(width: 448, height: 648), padding: 24)
        #expect(fit == 0.5)
        for zoom in [1.0, 1.25, 4.0] {
            let geometry = PageGeometry(imageSize: image, scale: fit * zoom)
            #expect(geometry.viewSize == CGSize(width: 400 * zoom, height: 600 * zoom))
            let point = CGPoint(x: 123, y: 456)
            #expect(isClose(geometry.toImage(geometry.toView(point)), point))
            #expect(geometry.toView(CGRect(x: 100, y: 200, width: 40, height: 20))
                == CGRect(x: 50 * zoom, y: 100 * zoom, width: 20 * zoom, height: 10 * zoom))
            #expect(geometry.toImage(CGSize(width: 10 * zoom, height: 5 * zoom)) == CGSize(width: 20, height: 10))
        }
    }

    @Test func hitTestingFollowsRotation() {
        let rect = CGRect(x: 100, y: 100, width: 200, height: 40)
        let nearRightEnd = CGPoint(x: 290, y: 120)
        #expect(BlockGeometry.contains(nearRightEnd, rect: rect, angle: 0))
        #expect(!BlockGeometry.contains(nearRightEnd, rect: rect, angle: 90))
        // Rotated a quarter turn clockwise, the box's right end points down the page.
        #expect(BlockGeometry.contains(CGPoint(x: 200, y: 210), rect: rect, angle: 90))
        #expect(isClose(BlockGeometry.position(of: CGPoint(x: 1, y: 0), rect: rect, angle: 90), CGPoint(x: 200, y: 220)))
    }

    @Test func resizeKeepsTheOppositeCornerFixed() {
        let rect = CGRect(x: 100, y: 100, width: 200, height: 80)
        let plain = BlockGeometry.resized(rect, angle: 0, handle: .bottomRight, by: CGSize(width: 20, height: -10))
        #expect(plain == CGRect(x: 100, y: 100, width: 220, height: 70))
        let fromLeft = BlockGeometry.resized(rect, angle: 0, handle: .left, by: CGSize(width: 30, height: 99))
        #expect(fromLeft == CGRect(x: 130, y: 100, width: 170, height: 80))

        for angle in [30.0, -75.0, 180.0] {
            for handle in ResizeHandle.allCases {
                let opposite = CGPoint(x: -handle.unit.x, y: -handle.unit.y)
                let before = BlockGeometry.position(of: opposite, rect: rect, angle: angle)
                let resized = BlockGeometry.resized(rect, angle: angle, handle: handle, by: CGSize(width: 17, height: -9))
                #expect(isClose(BlockGeometry.position(of: opposite, rect: resized, angle: angle), before))
            }
        }
        // The dragged corner follows the pointer.
        let delta = CGSize(width: 12, height: 25)
        let start = BlockGeometry.position(of: CGPoint(x: 1, y: 1), rect: rect, angle: 30)
        let dragged = BlockGeometry.resized(rect, angle: 30, handle: .bottomRight, by: delta)
        #expect(isClose(BlockGeometry.position(of: CGPoint(x: 1, y: 1), rect: dragged, angle: 30),
                        CGPoint(x: start.x + delta.width, y: start.y + delta.height)))
    }

    @Test func resizeStopsAtTheMinimumAndMoveStaysOnThePage() {
        let rect = CGRect(x: 100, y: 100, width: 50, height: 50)
        let tiny = BlockGeometry.resized(rect, angle: 0, handle: .right, by: CGSize(width: -500, height: 0))
        #expect(tiny.width == BlockGeometry.minimumSize && tiny.minX == 100)
        let moved = BlockGeometry.moved(rect, by: CGSize(width: -1000, height: 20), within: CGSize(width: 800, height: 1200))
        #expect(moved == CGRect(x: -25, y: 120, width: 50, height: 50))
    }
}

@Suite struct EditQueueTests {
    @Test func coalescesBehindTheRunningRequest() {
        var queue = EditQueue()
        let first = queue.enqueue(BlockEdit(index: 0, fontSize: 30), page: "a")
        #expect(first == EditQueue.Batch(page: "a", edits: [BlockEdit(index: 0, fontSize: 30)]))
        #expect(queue.enqueue(BlockEdit(index: 0, fontSize: 32), page: "a") == nil)
        #expect(queue.enqueue(BlockEdit(index: 1, translation: "x"), autofit: true, page: "a") == nil)
        #expect(queue.enqueue(BlockEdit(index: 0, bold: true), page: "a") == nil)
        #expect(queue.waitingEdits(page: "a").count == 2)
        // Another page is independent.
        #expect(queue.enqueue(BlockEdit(index: 0, italic: true), page: "b") != nil)

        let second = queue.finish(page: "a")
        #expect(second == EditQueue.Batch(
            page: "a", edits: [BlockEdit(index: 0, fontSize: 32, bold: true), BlockEdit(index: 1, translation: "x")], autofit: true))
        #expect(queue.isBusy(page: "a"))
        #expect(queue.enqueue(BlockEdit(index: 0, fontSize: 40), page: "a") == nil)
        #expect(queue.finish(page: "a")?.edits == [BlockEdit(index: 0, fontSize: 40)])
        #expect(queue.finish(page: "a") == nil)
        #expect(!queue.isBusy(page: "a"))
        #expect(queue.enqueue(BlockEdit(index: 0, fontSize: 41), page: "a") != nil)
    }

    @Test func deletionDropsQueuedEditsOfThatBlock() {
        var queue = EditQueue()
        _ = queue.enqueue(BlockEdit(index: 0, fontSize: 30), page: "a")
        _ = queue.enqueue(BlockEdit(index: 1, fontSize: 30), page: "a")
        _ = queue.enqueue(BlockEdit(index: 2, fontSize: 30), page: "a")
        #expect(queue.enqueueDeletion(of: 1, page: "a") == nil)
        let next = queue.finish(page: "a")
        #expect(next?.edits.map(\.index) == [2])
        #expect(next?.deletions == [1])
    }
}

/// Wraps the mock and records every method the model calls.
private actor RecordingEngine: Engine {
    let base: MockEngine
    private(set) var calls: [(method: String, params: JSONValue?)] = []
    nonisolated var events: AsyncStream<EngineEvent> { base.events }

    init(_ options: MockEngine.Options) { base = MockEngine(options: options) }
    func start() async throws { try await base.start() }
    func stop() async { await base.stop() }
    func diagnostics() async -> [String] { await base.diagnostics() }
    func call(_ method: String, params: JSONValue?) async throws -> JSONValue {
        calls.append((method, params))
        return try await base.call(method, params: params)
    }
    func count(of method: String) -> Int { calls.count { $0.method == method } }
    func params(of method: String) -> [JSONValue?] { calls.filter { $0.method == method }.map(\.params) }
}

@MainActor
private func waitUntil(_ condition: () -> Bool) async {
    for _ in 0..<500 where !condition() {
        try? await Task.sleep(for: .milliseconds(10))
    }
}

@MainActor
private func makeModel(translated: Int = 6, configure: (inout MockEngine.Options) -> Void = { _ in })
    async -> (AppModel, RecordingEngine) {
    var options = MockEngine.Options()
    options.stepDelay = .milliseconds(2)
    options.translatedPages = translated
    configure(&options)
    let engine = RecordingEngine(options)
    let model = AppModel(engine: engine)
    await model.start()
    await model.open([FileManager.default.temporaryDirectory])
    return (model, engine)
}

@MainActor
@Suite(.timeLimit(.minutes(1))) struct AppModelTests {
    @Test func openTranslateAndFinish() async throws {
        let (model, engine) = await makeModel(translated: 0) { $0.modelsReady = false }
        #expect(model.isEngineReady)
        #expect(model.project?.pages.count == 6)
        #expect(model.selectedPage == "001.png")
        let first = try #require(model.project?.pages.first)
        #expect(model.status(of: first) == .pending)

        await model.translate()
        #expect(model.run?.pages.count == 6)
        #expect(model.status(of: first) == .inProgress)
        await waitUntil { model.run?.prepare != nil }
        #expect(model.run?.prepare?.phase != nil)
        await waitUntil { model.run == nil && model.project?.pages.allSatisfy(\.done) == true }
        #expect(model.project?.pages.allSatisfy { $0.result != nil && $0.blocks.count == 3 } == true)
        #expect(model.project?.pages.allSatisfy { model.status(of: $0) == .done } == true)
        #expect(await engine.params(of: "run") == [["mode": "all"]])
    }

    @Test func translateContinuesWhenSomePagesAreDone() async {
        let (model, engine) = await makeModel(translated: 2)
        await model.translate()
        #expect(model.run?.pages == ["003.png", "004.png", "005.png", "006.png"])
        await waitUntil { model.run == nil }
        #expect(await engine.params(of: "run") == [["mode": "continue"]])
    }

    @Test func cancelShowsStoppingUntilTheRunEnds() async {
        let (model, _) = await makeModel(translated: 0) { $0.stepDelay = .milliseconds(30) }
        await model.translate()
        await model.cancelRun()
        #expect(model.run?.cancelling == true)
        await waitUntil { model.run == nil }
        #expect(model.run == nil)
        #expect(model.project?.pages.contains { !$0.done } == true)
    }

    @Test func failedPageIsMarkedAndReported() async throws {
        let (model, _) = await makeModel(translated: 0) { $0.failingPages = ["002.png"] }
        await model.translate()
        await waitUntil { model.run == nil }
        let failed = try #require(model.project?.pages.first { $0.name == "002.png" })
        #expect(model.status(of: failed) == .failed)
        #expect(model.notice?.message == "Translation failed for 002.png.")
    }

    @Test func translatorNotReadyOffersSettings() async {
        let (model, _) = await makeModel()
        await model.updateSettings(SettingsPatch(translator: .llm(profileID: "p1")))
        #expect(model.settings?.translator.kind == .llm)
        await model.translate()
        #expect(model.run == nil)
        #expect(model.notice?.action == .openTranslationSettings)
        #expect(model.settingsProfile == "p1")

        await model.updateProfile(id: "p1", ProfilePatch(apiKey: "test-key"))
        #expect(model.settings?.profiles.first?.hasAPIKey == true)
        await model.translate()
        #expect(model.run != nil)
        #expect(model.notice == nil)
        await waitUntil { model.run == nil }
    }

    @Test func editUpdatesThePageAndCanBeUndone() async throws {
        let (model, engine) = await makeModel()
        let undo = UndoManager()
        model.undoManager = undo
        let before = try #require(model.currentPage)
        let original = try #require(before.blocks.first)

        model.apply(BlockEdit(index: 0, translation: "你好", fontSize: 40), to: before.name)
        // Shown at once, before the engine answers.
        #expect(model.currentPage?.blocks.first?.translation == "你好")
        #expect(model.renderingPages == [before.name])
        await waitUntil { model.renderingPages.isEmpty }
        let after = try #require(model.currentPage)
        #expect(after.revision == before.revision + 1)
        #expect(after.blocks.first?.fontSize == 40)

        #expect(undo.canUndo)
        undo.undo()
        await waitUntil { model.renderingPages.isEmpty }
        #expect(model.currentPage?.blocks.first == original)
        #expect(model.currentPage?.revision == before.revision + 2)
        #expect(await engine.params(of: "update_blocks").last
            == ["page": "001.png", "edits": [["index": 0, "translation": .string(original.translation), "font_size": .number(original.fontSize)]]])

        #expect(undo.canRedo)
        undo.redo()
        await waitUntil { model.renderingPages.isEmpty }
        #expect(model.currentPage?.blocks.first?.translation == "你好")

        // An edit that changes nothing is not sent and not undoable twice.
        let calls = await engine.count(of: "update_blocks")
        model.apply(BlockEdit(index: 0, translation: "你好"), to: before.name)
        #expect(await engine.count(of: "update_blocks") == calls)
    }

    @Test func rapidEditsAreCoalescedAndNeverOverlap() async throws {
        let (model, engine) = await makeModel()
        let page = try #require(model.currentPage)
        let start = try #require(page.blocks.first?.rect)
        for step in 1...10 {
            model.apply(BlockEdit(index: 0, rect: start.offsetBy(dx: CGFloat(step), dy: 0)), to: page.name)
        }
        #expect(model.currentPage?.blocks.first?.rect == start.offsetBy(dx: 10, dy: 0))
        await waitUntil { model.renderingPages.isEmpty }
        // One request for the first nudge, one for the nine that arrived while it ran.
        #expect(await engine.count(of: "update_blocks") == 2)
        #expect(model.currentPage?.blocks.first?.rect == start.offsetBy(dx: 10, dy: 0))
        #expect(model.currentPage?.revision == page.revision + 2)
    }

    @Test func autofitUndoRestoresBoxAndSize() async throws {
        let (model, engine) = await makeModel()
        let undo = UndoManager()
        model.undoManager = undo
        let page = try #require(model.currentPage)
        let original = try #require(page.blocks.first)
        model.apply(BlockEdit(index: 0, rect: original.rect.offsetBy(dx: 50, dy: 50), fontSize: 60), to: page.name)
        await waitUntil { model.renderingPages.isEmpty }
        let moved = try #require(model.currentPage?.blocks.first)

        model.apply(BlockEdit(index: 0), autofit: true, to: page.name)
        await waitUntil { model.renderingPages.isEmpty }
        #expect(await engine.params(of: "update_blocks").last == ["page": "001.png", "edits": [["index": 0]], "autofit": true])
        #expect(model.currentPage?.blocks.first?.rect == original.rect)
        undo.undo()
        await waitUntil { model.renderingPages.isEmpty }
        #expect(model.currentPage?.blocks.first == moved)
    }

    @Test func deleteRemovesTheBlockAndClearsSelection() async throws {
        let (model, engine) = await makeModel()
        let undo = UndoManager()
        model.undoManager = undo
        let page = try #require(model.currentPage)
        model.selectedBlock = 1
        model.apply(BlockEdit(index: 1, bold: true), to: page.name)
        model.deleteBlock(1, on: page.name)
        #expect(model.selectedBlock == nil)
        #expect(!model.canEdit(page: page.name))
        await waitUntil { model.renderingPages.isEmpty }
        #expect(model.currentPage?.blocks.map(\.index) == [0, 2])
        #expect(await engine.params(of: "delete_blocks") == [["page": "001.png", "indexes": [1]]])
        #expect(model.canEdit(page: page.name))
        // Indexes may have shifted, so the earlier edit is no longer undoable.
        let updates = await engine.count(of: "update_blocks")
        if undo.canUndo { undo.undo() }
        await waitUntil { model.renderingPages.isEmpty }
        #expect(await engine.count(of: "update_blocks") == updates)
    }

    @Test func editsAreRefusedWhileTranslating() async throws {
        let (model, engine) = await makeModel { $0.stepDelay = .milliseconds(20) }
        let page = try #require(model.currentPage)
        await model.translate()
        model.apply(BlockEdit(index: 0, bold: true), to: page.name)
        #expect(await engine.count(of: "update_blocks") == 0)
        await model.cancelRun()
        await waitUntil { model.run == nil }
    }

    @Test func exportCopiesResults() async throws {
        let (model, _) = await makeModel(translated: 3)
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("export-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let files = try #require(await model.export(to: folder))
        #expect(files.count == 3)
        #expect(files.allSatisfy { FileManager.default.fileExists(atPath: $0) })
    }

    @Test func profileLifecycle() async throws {
        let (model, _) = await makeModel()
        let created = try #require(await model.createProfile(template: "OpenAI"))
        #expect(model.settings?.profiles.count == 2)
        await model.testProfile(id: created.id)
        #expect(model.profileTests[created.id] == .finished(ProfileTestResult(ok: false, message: "API key is missing.", kind: "auth", models: [])))
        await model.updateProfile(id: created.id, ProfilePatch(model: "model-small", apiKey: "k"))
        await model.testProfile(id: created.id)
        guard case .finished(let result) = model.profileTests[created.id] else {
            Issue.record("test did not finish")
            return
        }
        #expect(result.ok)
        #expect(model.settings?.profiles.last?.modelOptions == ["model-small", "model-large"])
        await model.deleteProfile(id: created.id)
        #expect(model.settings?.profiles.count == 1)
    }
}

@Suite struct ProjectImporterTests {
    @Test func copiesLooseImagesIntoANewFolder() async throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("import-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let source = base.appendingPathComponent("Chapter 1")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        let files = ["001.png", "002.jpg", "notes.txt"].map(source.appendingPathComponent)
        for file in files { try Data("x".utf8).write(to: file) }
        let projects = base.appendingPathComponent("Projects")

        let first = try await ProjectImporter.importImages(files, into: projects)
        #expect(first.lastPathComponent == "Chapter 1")
        #expect(try FileManager.default.contentsOfDirectory(atPath: first.path).sorted() == ["001.png", "002.jpg"])
        let second = try await ProjectImporter.importImages(files, into: projects)
        #expect(second.lastPathComponent == "Chapter 1 2")
        await #expect(throws: (any Error).self) {
            try await ProjectImporter.importImages([files[2]], into: projects)
        }
    }
}
