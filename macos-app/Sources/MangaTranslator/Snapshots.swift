import AppCore
import AppKit
import EngineKit
import SwiftUI

/// `--snapshot <dir>`: renders the main states against the mock engine into PNGs,
/// light and dark, without showing a window. For visual review, not a test.
@MainActor
enum Snapshots {
    static func run(directory: URL) -> Never {
        let application = NSApplication.shared
        application.setActivationPolicy(.prohibited)
        Task {
            do {
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try await renderAll(into: directory)
                exit(0)
            } catch {
                FileHandle.standardError.write(Data("snapshot failed: \(error)\n".utf8))
                exit(1)
            }
        }
        application.run()
        exit(0)
    }

    private static func renderAll(into directory: URL) async throws {
        let window = CGSize(width: 1120, height: 760)
        let settings = CGSize(width: 620, height: 470)
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("MangaTranslatorSnapshots/Chapter 12")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        func model(_ configure: (inout MockEngine.Options) -> Void = { _ in }) async -> AppModel {
            var options = MockEngine.Options()
            options.translatedPages = 6
            configure(&options)
            let model = AppModel(engine: MockEngine(options: options))
            await model.start()
            return model
        }
        func wait(_ condition: () -> Bool) async {
            for _ in 0..<400 where !condition() {
                try? await Task.sleep(for: .milliseconds(25))
            }
        }

        let empty = await model()
        try await capture(ContentView().environment(empty), "empty", window, into: directory)

        let broken = AppModel(engine: EngineClient { throw EngineError.launchFailed("Python not found at /opt/engine/.venv/bin/python.") })
        await broken.start()
        try await capture(ContentView().environment(broken), "engine-failed", window, into: directory)

        let main = await model()
        await main.open([folder])
        main.selectedBlock = 0
        try await capture(ContentView().environment(main), "main", window, into: directory)
        // Lists and forms inside the split view's glass columns do not draw offscreen; show them alone.
        let pane = CGSize(width: 280, height: 620)
        try await capture(SidebarView().environment(main), "pane-sidebar", pane, into: directory)
        try await capture(InspectorView().environment(main), "pane-inspector", pane, into: directory)

        // A page with withheld blocks, one of them selected, then the same jump on a zoomed page.
        main.selectNextReview()
        try await capture(ContentView().environment(main), "review", window, into: directory)
        try await capture(InspectorView().environment(main), "pane-inspector-review",
                          CGSize(width: pane.width, height: 800), into: directory)
        main.zoom = 2.5
        main.selectNextReview()
        try await capture(ContentView().environment(main), "review-zoomed", window, into: directory)

        let untranslated = await model { $0.translatedPages = 0 }
        await untranslated.open([folder])
        try await capture(ContentView().environment(untranslated), "not-translated", window, into: directory)

        let translating = await model {
            $0.translatedPages = 2
            $0.stepDelay = .seconds(1)
        }
        await translating.open([folder])
        await translating.translate()
        await wait { translating.run?.stage == "detect" && translating.run?.done == 2 }
        try await capture(ContentView().environment(translating), "translating", window, into: directory)
        try await capture(SidebarView().environment(translating), "pane-sidebar-translating", pane, into: directory)

        // A long step keeps the run in "Stopping…" while both appearances are captured.
        let stopping = await model {
            $0.translatedPages = 2
            $0.stepDelay = .seconds(8)
        }
        await stopping.open([folder])
        await stopping.translate()
        await stopping.cancelRun()
        try await capture(ContentView().environment(stopping), "stopping", window, into: directory)

        let preparing = await model {
            $0.translatedPages = 0
            $0.modelsReady = false
            $0.stepDelay = .seconds(1)
        }
        await preparing.open([folder])
        await preparing.translate()
        await wait { (preparing.run?.prepare?.done ?? 0) > 50_000_000 }
        try await capture(ContentView().environment(preparing), "preparing", window, into: directory)

        let notReady = await model()
        await notReady.open([folder])
        await notReady.updateSettings(SettingsPatch(translator: .llm(profileID: "p1")))
        await notReady.translate()
        try await capture(ContentView().environment(notReady), "error-translator", window, into: directory)

        let failed = await model {
            $0.translatedPages = 0
            $0.stepDelay = .milliseconds(5)
            $0.failingPages = ["003.png"]
        }
        await failed.open([folder])
        await failed.translate()
        await wait { failed.run == nil }
        failed.selectedPage = "003.png"
        try await capture(ContentView().environment(failed), "error-page", window, into: directory)
        try await capture(SidebarView().environment(failed), "pane-sidebar-failed", pane, into: directory)

        let services = await model()
        services.settingsProfile = "p1"
        await services.testProfile(id: "p1")
        try await capture(TranslationSettings().environment(services), "settings-translation", settings, into: directory)
        try await capture(OutputSettingsPane().environment(services), "settings-output", settings, into: directory)
        try await capture(EngineSettingsPane().environment(services), "settings-engine", settings, into: directory)
    }

    private static func capture(_ view: some View, _ name: String, _ size: CGSize, into directory: URL) async throws {
        for (suffix, appearance) in [("light", NSAppearance.Name.aqua), ("dark", .darkAqua)] {
            let window = NSWindow(contentRect: CGRect(origin: .zero, size: size),
                                  styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
                                  backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            // App-wide, so split-view columns and sheets follow too.
            NSApp.appearance = NSAppearance(named: appearance)
            let host = NSHostingView(rootView: view)
            host.sceneBridgingOptions = .all
            window.contentView = host
            window.setContentSize(size)
            window.layoutIfNeeded()
            // Let async image loads and layout settle.
            try await Task.sleep(for: .milliseconds(900))
            window.displayIfNeeded()

            // The theme frame includes the title bar and toolbar when they exist.
            let target = host.superview ?? host
            guard let bitmap = target.bitmapImageRepForCachingDisplay(in: target.bounds) else { continue }
            target.cacheDisplay(in: target.bounds, to: bitmap)
            guard let png = bitmap.representation(using: .png, properties: [:]) else { continue }
            try png.write(to: directory.appendingPathComponent("\(name)-\(suffix).png"))
            window.close()
        }
    }
}
