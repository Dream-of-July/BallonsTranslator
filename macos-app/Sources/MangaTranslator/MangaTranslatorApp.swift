import AppCore
import AppKit
import EngineKit
import SwiftUI

@main
enum Main {
    @MainActor
    static func main() {
        let arguments = CommandLine.arguments
        if arguments.contains("--selfcheck") {
            exit(selfCheck())
        }
        if let flag = arguments.firstIndex(of: "--snapshot"), arguments.indices.contains(flag + 1) {
            Snapshots.run(directory: URL(fileURLWithPath: arguments[flag + 1]))
        }
        MangaTranslatorApp.main()
    }

    /// `--selfcheck`: no window, no engine process. Prints what the app would launch.
    private static func selfCheck() -> Int32 {
        print("protocol \(engineProtocolVersion)")
        do {
            let location = try EngineLocator.resolve()
            print("engine root \(location.root.path)")
            print("engine python \(location.python.path)")
        } catch {
            print("engine not found: \(error.localizedDescription)")
        }
        return 0
    }
}

@MainActor
enum AppEnvironment {
    /// `--mock` runs against the in-process mock engine and remembers nothing.
    static let model: AppModel = {
        if CommandLine.arguments.contains("--mock") {
            var options = MockEngine.Options()
            options.modelsReady = false
            options.translatedPages = 2
            return AppModel(engine: MockEngine(options: options))
        }
        let client = EngineClient { EngineLaunch(location: try EngineLocator.resolve()) }
        return AppModel(engine: client, defaults: .standard)
    }()
}

struct MangaTranslatorApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    private let model = AppEnvironment.model

    var body: some Scene {
        Window("Manga Translator", id: "main") {
            ContentView()
                .environment(model)
        }
        .defaultSize(width: 1120, height: 780)
        .commands {
            AppCommands(model: model)
        }

        Settings {
            SettingsView()
                .environment(model)
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        Task { await AppEnvironment.model.start() }
    }

    /// Folders or images dropped on the Dock icon.
    func application(_ application: NSApplication, open urls: [URL]) {
        Task { await AppEnvironment.model.open(urls) }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

/// File panels shared by the toolbar, the menu bar and the start screen.
@MainActor
enum FileActions {
    static func open(_ model: AppModel) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = [.folder, .image]
        present(panel) { urls in
            await model.open(urls)
        }
    }

    static func export(_ model: AppModel) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = String(localized: "Export")
        panel.message = String(localized: "Choose where to save the translated pages.")
        present(panel) { urls in
            guard let folder = urls.first, let files = await model.export(to: folder), !files.isEmpty else { return }
            NSWorkspace.shared.activateFileViewerSelecting(files.map { URL(fileURLWithPath: $0) })
        }
    }

    private static func present(_ panel: NSOpenPanel, then action: @escaping @MainActor ([URL]) async -> Void) {
        let completion: (NSApplication.ModalResponse) -> Void = { response in
            guard response == .OK else { return }
            let urls = panel.urls
            Task { await action(urls) }
        }
        if let window = NSApp.keyWindow {
            panel.beginSheetModal(for: window, completionHandler: completion)
        } else {
            panel.begin(completionHandler: completion)
        }
    }
}

struct AppCommands: Commands {
    let model: AppModel
    @AppStorage("showsInspector") private var showsInspector = true

    var body: some Commands {
        @Bindable var model = model

        CommandGroup(replacing: .newItem) {
            Button("Open…") { FileActions.open(model) }
                .keyboardShortcut("o")
            Button("Export…") { FileActions.export(model) }
                .keyboardShortcut("e")
                .disabled(model.project == nil)
        }
        SidebarCommands()
        CommandGroup(after: .sidebar) {
            Toggle("Show Inspector", isOn: $showsInspector)
                .keyboardShortcut("i", modifiers: [.command, .option])
            Divider()
            Toggle("Show Original", isOn: $model.showsOriginal)
                .keyboardShortcut("o", modifiers: [.command, .shift])
                .disabled(model.currentPage?.result == nil)
            Button("Zoom In") { model.zoomIn() }
                .keyboardShortcut("+")
            Button("Zoom Out") { model.zoomOut() }
                .keyboardShortcut("-")
            Button("Zoom to Fit") { model.zoomToFit() }
                .keyboardShortcut("0")
            Divider()
        }
        CommandMenu("Translate") {
            Button("Translate") { Task { await model.translate() } }
                .keyboardShortcut("r")
                .disabled(!model.canTranslate)
            Button("Translate This Page") {
                if let page = model.selectedPage { Task { await model.translate(pages: [page]) } }
            }
            .disabled(!model.canTranslate || model.selectedPage == nil)
            Button("Stop") { Task { await model.cancelRun() } }
                .keyboardShortcut(".")
                .disabled(model.run == nil || model.run?.cancelling == true)
            Divider()
            Button("Next Item to Review") {
                showsInspector = true
                model.selectNextReview()
            }
            .keyboardShortcut("'")
            .disabled(model.reviewTotal == 0)
        }
    }
}
