import Foundation
import Testing
@testable import EngineKit

/// Decodes what the real Python engine sends. Opt-in: needs an engine checkout and an
/// already-translated project, so it is skipped unless both variables are set.
///
///     MANGA_ENGINE_ROOT=<repo> MANGA_ENGINE_PROJECT=<translated folder> swift test --filter RealEngine
@Suite(.enabled(if: ProcessInfo.processInfo.environment["MANGA_ENGINE_PROJECT"] != nil))
struct RealEngineTests {
    @Test func roundTripsAgainstTheRealEngine() async throws {
        let environment = ProcessInfo.processInfo.environment
        let location = try EngineLocator.resolve(settings: [:], environment: environment)
        let source = URL(fileURLWithPath: try #require(environment["MANGA_ENGINE_PROJECT"]))

        // Work on a copy so the fixture project is never modified.
        let work = FileManager.default.temporaryDirectory.appendingPathComponent("engine-it-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: work, withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: source, to: work.appendingPathComponent(source.lastPathComponent))
        defer { try? FileManager.default.removeItem(at: work) }
        let projectPath = work.appendingPathComponent(source.lastPathComponent).path

        let engine = EngineClient(launch: { EngineLaunch(location: location) })
        try await engine.start()
        defer { Task { await engine.stop() } }

        let hello = try await engine.hello()
        #expect(hello.protocolVersion == 1)

        let project = try await engine.openProject(path: projectPath)
        let page = try #require(project.pages.first)
        let block = try #require(page.blocks.first)
        #expect(page.width > 0 && page.result != nil)

        let fonts = try await engine.listFonts()
        #expect(!fonts.fonts.isEmpty && !fonts.default.isEmpty)

        let settings = try await engine.settings()
        #expect(!settings.languages.target.isEmpty && !settings.providerTemplates.isEmpty)

        var moved = block.rect
        moved.origin.x += 20
        let edited = try await engine.updateBlocks(
            page: page.name,
            edits: [BlockEdit(index: block.index, translation: "集成测试", rect: moved, fontSize: block.fontSize + 6)]
        )
        let after = try #require(edited.blocks.first { $0.index == block.index })
        #expect(edited.revision > page.revision)
        #expect(after.translation == "集成测试")
        #expect(abs(after.rect.origin.x - moved.origin.x) < 1)
        #expect(abs(after.fontSize - (block.fontSize + 6)) < 0.5)

        let exported = try await engine.export(to: work.appendingPathComponent("out").path, what: .images)
        #expect(exported.count == project.pages.filter { $0.result != nil }.count)
    }
}
