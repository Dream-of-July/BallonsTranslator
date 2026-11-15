import AppCore
import AppKit
import EngineKit
import SwiftUI

struct SettingsView: View {
    enum Pane: String {
        case translation, output, engine
    }

    static let paneKey = "settingsPane"

    @AppStorage(SettingsView.paneKey) private var pane = Pane.translation

    var body: some View {
        TabView(selection: $pane) {
            Tab("Translation", systemImage: "translate", value: .translation) {
                TranslationSettings()
            }
            Tab("Output", systemImage: "photo", value: .output) {
                OutputSettingsPane()
            }
            Tab("Engine", systemImage: "gearshape.2", value: .engine) {
                EngineSettingsPane()
            }
        }
        .frame(width: 620, height: 420)
    }
}

// MARK: - Translation

struct TranslationSettings: View {
    @Environment(AppModel.self) private var model
    /// nil is Google.
    @State private var selection: String?

    var body: some View {
        if let settings = model.settings {
            HStack(spacing: 0) {
                serviceList(settings)
                    .frame(width: 190)
                Divider()
                Group {
                    if let profile = settings.profiles.first(where: { $0.id == selection }) {
                        ProfileForm(profile: profile, isActive: settings.translator.profileID == profile.id)
                            .id(profile.id)
                    } else {
                        googleForm(settings)
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
            .onAppear {
                selection = model.settingsProfile ?? settings.translator.profileID
                model.settingsProfile = nil
            }
            .onChange(of: model.settingsProfile) {
                if let profile = model.settingsProfile {
                    selection = profile
                    model.settingsProfile = nil
                }
            }
        } else {
            ContentUnavailableView("Engine Not Running", systemImage: "bolt.slash")
        }
    }

    private func serviceList(_ settings: EngineSettings) -> some View {
        List(selection: $selection) {
            ServiceRow(name: "Google", isActive: settings.translator.kind == .google)
                .tag(String?.none)
            ForEach(settings.profiles) { profile in
                ServiceRow(name: profile.name, isActive: settings.translator.profileID == profile.id)
                    .tag(Optional(profile.id))
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            HStack(spacing: 2) {
                Menu("Add Service", systemImage: "plus") {
                    ForEach(settings.providerTemplates, id: \.self) { template in
                        Button(template) {
                            Task { selection = await model.createProfile(template: template)?.id ?? selection }
                        }
                    }
                }
                .menuIndicator(.hidden)
                .fixedSize()
                Button("Remove Service", systemImage: "minus") {
                    if let id = selection {
                        selection = nil
                        Task { await model.deleteProfile(id: id) }
                    }
                }
                .disabled(selection == nil)
                Spacer()
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.borderless)
            .padding(8)
        }
    }

    private func googleForm(_ settings: EngineSettings) -> some View {
        Form {
            Section {
                LabeledContent("Service", value: "Google")
                Text("Works without an account.")
                    .foregroundStyle(.secondary)
            }
            if settings.translator.kind != .google {
                Button("Use for Translation") {
                    Task { await model.updateSettings(SettingsPatch(translator: .google)) }
                }
            }
        }
        .formStyle(.grouped)
    }
}

private struct ServiceRow: View {
    var name: String
    var isActive: Bool

    var body: some View {
        HStack {
            Text(verbatim: name)
                .lineLimit(1)
            Spacer()
            if isActive {
                Image(systemName: "checkmark")
                    .foregroundStyle(.secondary)
                    .accessibilityLabel("In use")
            }
        }
    }
}

private struct ProfileForm: View {
    private enum Field {
        case name, baseURL, model, apiKey
    }

    @Environment(AppModel.self) private var model
    var profile: Profile
    var isActive: Bool
    @State private var name: String
    @State private var baseURL: String
    @State private var modelName: String
    @State private var apiKey = ""
    @FocusState private var focus: Field?

    init(profile: Profile, isActive: Bool) {
        self.profile = profile
        self.isActive = isActive
        _name = State(initialValue: profile.name)
        _baseURL = State(initialValue: profile.baseURL)
        _modelName = State(initialValue: profile.model)
    }

    var body: some View {
        Form {
            Section {
                TextField("Name", text: $name)
                    .focused($focus, equals: .name)
                TextField("Base URL", text: $baseURL, prompt: Text(verbatim: "https://…/v1"))
                    .focused($focus, equals: .baseURL)
                    .autocorrectionDisabled()
                LabeledContent("Model") {
                    HStack(spacing: 4) {
                        TextField("Model", text: $modelName)
                            .labelsHidden()
                            .focused($focus, equals: .model)
                            .autocorrectionDisabled()
                        if !profile.modelOptions.isEmpty {
                            Menu("Choose Model", systemImage: "chevron.up.chevron.down") {
                                ForEach(profile.modelOptions, id: \.self) { option in
                                    Button(option) {
                                        modelName = option
                                        save()
                                    }
                                }
                            }
                            .labelStyle(.iconOnly)
                            .menuIndicator(.hidden)
                            .fixedSize()
                        }
                    }
                }
                LabeledContent("API Key") {
                    HStack(spacing: 6) {
                        SecureField("API Key", text: $apiKey, prompt: Text(keyPrompt))
                            .labelsHidden()
                            .focused($focus, equals: .apiKey)
                        if profile.hasAPIKey {
                            Button("Remove Key", systemImage: "xmark.circle.fill") {
                                apiKey = ""
                                Task { await model.updateProfile(id: profile.id, ProfilePatch(apiKey: "")) }
                            }
                            .labelStyle(.iconOnly)
                            .buttonStyle(.borderless)
                            .help("Remove the saved key")
                        }
                    }
                }
            }
            Section {
                HStack {
                    Button("Test Connection") {
                        save()
                        Task { await model.testProfile(id: profile.id) }
                    }
                    .disabled(model.profileTests[profile.id] == .running)
                    Spacer()
                    if !isActive {
                        Button("Use for Translation") {
                            save()
                            Task { await model.updateSettings(SettingsPatch(translator: .llm(profileID: profile.id))) }
                        }
                    }
                }
                testResult
            }
        }
        .formStyle(.grouped)
        .onSubmit(save)
        .onChange(of: focus) { save() }
        .onDisappear(perform: save)
    }

    private var keyPrompt: LocalizedStringKey {
        profile.hasAPIKey ? "Saved" : profile.requireAPIKey ? "Required" : "Optional"
    }

    @ViewBuilder private var testResult: some View {
        switch model.profileTests[profile.id] {
        case .running:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Testing…").foregroundStyle(.secondary)
            }
        case .finished(let result):
            Label {
                Text(verbatim: result.message)
            } icon: {
                Image(systemName: result.ok ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(result.ok ? .green : .red)
            }
            .lineLimit(3)
        case nil:
            EmptyView()
        }
    }

    /// Sends whatever differs from the engine's copy. The key is write-only, so it is
    /// sent once and the field cleared; "Saved" then comes from `has_api_key`.
    private func save() {
        var patch = ProfilePatch()
        if name != profile.name, !name.isEmpty { patch.name = name }
        if baseURL != profile.baseURL { patch.baseURL = baseURL }
        if modelName != profile.model { patch.model = modelName }
        if !apiKey.isEmpty {
            patch.apiKey = apiKey
            apiKey = ""
        }
        guard !patch.isEmpty else { return }
        let id = profile.id
        Task { await model.updateProfile(id: id, patch) }
    }
}

// MARK: - Output

struct OutputSettingsPane: View {
    @Environment(AppModel.self) private var model
    @State private var quality = 95.0

    var body: some View {
        if let output = model.settings?.output {
            Form {
                Picker("Format", selection: Binding { output.format } set: { format in
                    Task { await model.updateSettings(SettingsPatch(output: OutputSettings(format: format, quality: output.quality))) }
                }) {
                    Text(verbatim: "PNG").tag(".png")
                    Text(verbatim: "JPEG").tag(".jpg")
                    Text(verbatim: "WebP").tag(".webp")
                }
                LabeledContent("Quality") {
                    HStack(spacing: 8) {
                        Slider(value: $quality, in: 1...100, step: 1) { editing in
                            if !editing, Int(quality) != output.quality {
                                Task { await model.updateSettings(SettingsPatch(output: OutputSettings(format: output.format, quality: Int(quality)))) }
                            }
                        }
                        .labelsHidden()
                        .accessibilityLabel("Quality")
                        Text(verbatim: "\(Int(quality))")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                            .frame(width: 28, alignment: .trailing)
                    }
                }
                .disabled(output.format == ".png")
            }
            .formStyle(.grouped)
            .onChange(of: output.quality, initial: true) { quality = Double(output.quality) }
        } else {
            ContentUnavailableView("Engine Not Running", systemImage: "bolt.slash")
        }
    }
}

// MARK: - Engine

struct EngineSettingsPane: View {
    @Environment(AppModel.self) private var model
    @AppStorage(EngineLocator.rootDefaultsKey) private var root = ""
    @AppStorage(EngineLocator.pythonDefaultsKey) private var python = ""
    @State private var showsLog = false

    var body: some View {
        Form {
            Section {
                LabeledContent("Status") { status }
                if case .ready(let hello) = model.engineState {
                    LabeledContent("Version", value: hello.engineVersion)
                    LabeledContent("Device", value: device(hello))
                    if !hello.modelsReady {
                        LabeledContent("Models", value: String(localized: "Downloaded on first use"))
                    }
                }
            }
            Section("Location") {
                PathField(title: "Engine Folder", path: $root, prompt: "Automatic", directories: true)
                PathField(title: "Python", path: $python, prompt: "Automatic", directories: false)
            }
            Section {
                HStack {
                    Button("Show Log") { showsLog = true }
                    Spacer()
                    Button("Restart Engine") { Task { await model.restartEngine() } }
                        .disabled(model.engineState == .starting)
                }
            }
        }
        .formStyle(.grouped)
        .sheet(isPresented: $showsLog) {
            DiagnosticsView()
        }
    }

    @ViewBuilder private var status: some View {
        switch model.engineState {
        case .starting:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Starting…")
            }
        case .ready:
            Label("Running", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
        case .failed(let message):
            Label(message, systemImage: "exclamationmark.triangle.fill")
                .symbolRenderingMode(.multicolor)
        }
    }

    private func device(_ hello: Hello) -> String {
        switch hello.device {
        case "mps": String(localized: "Apple GPU")
        case "cuda": "CUDA"
        case "cpu": "CPU"
        case let other?: other
        case nil: hello.torchInstalled ? "CPU" : String(localized: "Not installed yet")
        }
    }
}

private struct PathField: View {
    var title: LocalizedStringKey
    @Binding var path: String
    var prompt: LocalizedStringKey
    var directories: Bool

    var body: some View {
        LabeledContent(title) {
            HStack(spacing: 6) {
                TextField(title, text: $path, prompt: Text(prompt))
                    .labelsHidden()
                    .autocorrectionDisabled()
                Button("Choose…") {
                    let panel = NSOpenPanel()
                    panel.canChooseDirectories = directories
                    panel.canChooseFiles = !directories
                    panel.showsHiddenFiles = true
                    if panel.runModal() == .OK, let url = panel.url {
                        path = url.path
                    }
                }
            }
        }
    }
}

// MARK: - Log

struct DiagnosticsView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var lines: [String] = []

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                Text(verbatim: lines.isEmpty ? " " : lines.joined(separator: "\n"))
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }
            .defaultScrollAnchor(.bottom)
            Divider()
            HStack {
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(lines.joined(separator: "\n"), forType: .string)
                }
                .disabled(lines.isEmpty)
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(12)
        }
        .frame(width: 640, height: 420)
        .task {
            // Refreshes while open; the engine keeps writing.
            while !Task.isCancelled {
                lines = await model.logLines()
                try? await Task.sleep(for: .seconds(1))
            }
        }
    }
}
