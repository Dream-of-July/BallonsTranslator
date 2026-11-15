import AppCore
import EngineKit
import SwiftUI

struct ContentView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.undoManager) private var undoManager
    @State private var isDropTarget = false

    var body: some View {
        Group {
            if model.project == nil {
                StartView(isDropTarget: isDropTarget)
            } else {
                ProjectView()
            }
        }
        .frame(minWidth: 780, minHeight: 500)
        .dropDestination(for: URL.self) { urls, _ in
            Task { await model.open(urls) }
            return true
        } isTargeted: {
            isDropTarget = $0
        }
        .onChange(of: undoManager, initial: true) {
            model.undoManager = undoManager
        }
    }
}

// MARK: - No project

struct StartView: View {
    @Environment(AppModel.self) private var model
    var isDropTarget: Bool
    @State private var showsLog = false

    var body: some View {
        ContentUnavailableView {
            Label("Drop a folder of pages", systemImage: "photo.on.rectangle.angled")
        } actions: {
            Button("Open…") { FileActions.open(model) }
                .buttonStyle(.glassProminent)
                .controlSize(.large)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .overlay {
            if isDropTarget {
                RoundedRectangle(cornerRadius: 16)
                    .strokeBorder(.tint, style: StrokeStyle(lineWidth: 2, dash: [8, 6]))
                    .padding(16)
            }
        }
        .safeAreaInset(edge: .bottom) {
            engineStatus
                .font(.callout)
                .padding(.bottom, 20)
        }
        .sheet(isPresented: $showsLog) {
            DiagnosticsView()
        }
    }

    @ViewBuilder private var engineStatus: some View {
        switch model.engineState {
        case .starting:
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text(model.isOpening ? "Opening…" : "Starting engine…")
            }
            .foregroundStyle(.secondary)
        case .failed(let message):
            EngineFailure(message: message, showsLog: $showsLog)
        case .ready:
            if model.isOpening {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Opening…")
                }
                .foregroundStyle(.secondary)
            } else if let notice = model.notice {
                Label(notice.message, systemImage: "exclamationmark.triangle.fill")
                    .symbolRenderingMode(.multicolor)
            }
        }
    }
}

/// "Engine failed" message with the two ways out: restart, or look at the log.
struct EngineFailure: View {
    @Environment(AppModel.self) private var model
    var message: String
    @Binding var showsLog: Bool

    var body: some View {
        HStack(spacing: 10) {
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .symbolRenderingMode(.multicolor)
                .lineLimit(2)
            Spacer(minLength: 12)
            Button("Show Log") { showsLog = true }
            Button("Restart Engine") { Task { await model.restartEngine() } }
        }
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: 640)
        .padding(.horizontal, 16)
    }
}

// MARK: - Project

struct ProjectView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("showsInspector") private var showsInspector = true

    var body: some View {
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 180, ideal: 220, max: 320)
        } detail: {
            PageDetail()
                .inspector(isPresented: $showsInspector) {
                    InspectorView()
                        .inspectorColumnWidth(min: 250, ideal: 280, max: 380)
                }
        }
        .navigationTitle(model.project?.name ?? "")
        .navigationSubtitle(subtitle)
        .toolbar {
            ToolbarItem {
                Button("Open", systemImage: "folder") { FileActions.open(model) }
                    .help("Open a folder of pages")
            }
            ToolbarItem {
                LanguageMenu()
            }
            ToolbarItem {
                TranslatorMenu()
            }
            ToolbarItem(placement: .primaryAction) {
                TranslateButton()
            }
            ToolbarSpacer(.fixed)
            ToolbarItem(placement: .primaryAction) {
                Button("Export", systemImage: "square.and.arrow.up") { FileActions.export(model) }
                    .disabled(model.project?.pages.contains { $0.result != nil } != true)
                    .help("Export translated pages")
            }
            ToolbarItem(placement: .primaryAction) {
                Button("Inspector", systemImage: "sidebar.trailing") { showsInspector.toggle() }
                    .help(showsInspector ? "Hide Inspector" : "Show Inspector")
            }
        }
    }

    private var subtitle: String {
        guard let project = model.project else { return "" }
        if case .starting = model.engineState { return String(localized: "Starting engine…") }
        let done = model.translatedPageCount
        let review = model.reviewTotal
        if review > 0 {
            return String(localized: "\(done) of \(project.pages.count) pages translated · \(review) to review")
        }
        return String(localized: "\(done) of \(project.pages.count) pages translated")
    }
}

struct TranslateButton: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let run = model.run {
            Button(run.cancelling ? "Stopping…" : "Stop", systemImage: "stop.fill") {
                Task { await model.cancelRun() }
            }
            .labelStyle(.titleAndIcon)
            .disabled(run.cancelling)
            .help("Stop translating")
        } else {
            Button("Translate", systemImage: "translate") {
                Task { await model.translate() }
            }
            .labelStyle(.titleAndIcon)
            .buttonStyle(.borderedProminent)
            .disabled(!model.canTranslate)
            .help("Translate all pages")
        }
    }
}

struct LanguageMenu: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        Menu {
            if let settings = model.settings {
                Picker("From", selection: binding(settings.sourceLanguage) { SettingsPatch(sourceLanguage: $0) }) {
                    ForEach(settings.languages.source, id: \.self) { Text(verbatim: $0) }
                }
                Picker("To", selection: binding(settings.targetLanguage) { SettingsPatch(targetLanguage: $0) }) {
                    ForEach(settings.languages.target, id: \.self) { Text(verbatim: $0) }
                }
            }
        } label: {
            if let settings = model.settings {
                Text(verbatim: "\(settings.sourceLanguage) → \(settings.targetLanguage)")
            } else {
                Text("Languages")
            }
        }
        .pickerStyle(.inline)
        .disabled(model.settings == nil || model.run != nil)
        .help("Languages")
    }

    private func binding(_ value: String, patch: @escaping (String) -> SettingsPatch) -> Binding<String> {
        Binding { value } set: { newValue in
            Task { await model.updateSettings(patch(newValue)) }
        }
    }
}

struct TranslatorMenu: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openSettings) private var openSettings

    private static let google = "\u{0}google"

    var body: some View {
        Menu {
            if let settings = model.settings {
                Picker("Service", selection: selection(settings)) {
                    Text(verbatim: "Google").tag(Self.google)
                    ForEach(settings.profiles) { profile in
                        Text(verbatim: profile.name).tag(profile.id)
                    }
                }
                .pickerStyle(.inline)
                Divider()
            }
            Button("Add Service…") {
                UserDefaults.standard.set(SettingsView.Pane.translation.rawValue, forKey: SettingsView.paneKey)
                openSettings()
            }
        } label: {
            Text(verbatim: title)
        }
        .disabled(model.settings == nil || model.run != nil)
        .help("Translation service")
    }

    private var title: String {
        guard let settings = model.settings else { return "Google" }
        switch settings.translator.kind {
        case .google: return "Google"
        case .llm: return settings.profiles.first { $0.id == settings.translator.profileID }?.name ?? settings.translator.module
        case .other: return settings.translator.module
        }
    }

    private func selection(_ settings: EngineSettings) -> Binding<String> {
        Binding {
            switch settings.translator.kind {
            case .google: Self.google
            case .llm: settings.translator.profileID ?? ""
            case .other: ""
            }
        } set: { id in
            Task { await model.updateSettings(SettingsPatch(translator: id == Self.google ? .google : .llm(profileID: id))) }
        }
    }
}

// MARK: - Sidebar

struct SidebarView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        List(selection: $model.selectedPage) {
            ForEach(Array((model.project?.pages ?? []).enumerated()), id: \.element.name) { offset, page in
                PageRow(page: page, number: offset + 1, status: model.status(of: page), review: model.reviewCount(of: page))
                    .tag(page.name)
                    .contextMenu {
                        Button("Translate This Page") {
                            Task { await model.translate(pages: [page.name]) }
                        }
                        .disabled(!model.canTranslate)
                    }
            }
        }
    }
}

struct PageRow: View {
    var page: Page
    var number: Int
    var status: AppModel.PageStatus
    /// Blocks left to review; shown in place of the check mark on a translated page.
    var review: Int

    var body: some View {
        HStack(spacing: 10) {
            Thumbnail(path: page.result ?? page.source, revision: page.revision)
                .frame(width: 40, height: 56)
            Text(verbatim: (page.name as NSString).deletingPathExtension)
                .lineLimit(1)
                .truncationMode(.middle)
            Spacer(minLength: 4)
            statusMark
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Page \(number), \(statusText)"))
    }

    @ViewBuilder private var statusMark: some View {
        switch status {
        case .pending:
            Image(systemName: "circle.dotted").foregroundStyle(.tertiary)
        case .inProgress:
            ProgressView().controlSize(.small)
        case .done where review > 0:
            Text(verbatim: "\(review)")
                .font(.caption.weight(.semibold))
                .monospacedDigit()
                .foregroundStyle(.white)
                .padding(.horizontal, 5)
                .frame(minWidth: 17, minHeight: 17)
                .background(.orange, in: .capsule)
        case .done:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill").symbolRenderingMode(.multicolor)
        }
    }

    private var statusText: String {
        switch status {
        case .pending: String(localized: "not translated")
        case .inProgress: String(localized: "translating")
        case .done where review > 0: String(localized: "translated, \(review) to review")
        case .done: String(localized: "translated")
        case .failed: String(localized: "failed")
        }
    }
}

struct Thumbnail: View {
    var path: String
    var revision: Int
    @State private var image: CGImage?

    var body: some View {
        ZStack {
            if let image {
                Image(decorative: image, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            } else {
                Rectangle().fill(.quaternary)
            }
        }
        .clipShape(.rect(cornerRadius: 3))
        .task(id: "\(path)#\(revision)") {
            if let loaded = await ImageLoader.load(path, maxPixelSize: 160) {
                image = loaded
            }
        }
    }
}
