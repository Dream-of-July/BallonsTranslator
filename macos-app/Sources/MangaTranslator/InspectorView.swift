import AppCore
import EngineKit
import SwiftUI

struct InspectorView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if let page = model.currentPage, let block = model.currentBlock {
            BlockForm(page: page.name, block: block)
                .id("\(page.name)#\(block.index)")
                .disabled(!model.canEdit(page: page.name))
        } else {
            ContentUnavailableView("Select a bubble to edit", systemImage: "text.bubble")
        }
    }
}

/// Editor for one text block. Every change goes to `AppModel.apply`, which shows it at
/// once and has the engine redraw the page.
private struct BlockForm: View {
    @Environment(AppModel.self) private var model
    var page: String
    var block: Block
    @State private var draft: String
    @State private var sourceDraft: String
    @State private var color: Color
    @FocusState private var focus: Field?

    private enum Field {
        case source, translation
    }

    init(page: String, block: Block) {
        self.page = page
        self.block = block
        _draft = State(initialValue: block.translation)
        _sourceDraft = State(initialValue: block.sourceText)
        _color = State(initialValue: Color(block.color))
    }

    var body: some View {
        Form {
            // Own strings table: "Original" and "Translation" mean other things elsewhere in the app.
            if block.needsReview {
                Section {
                    Label {
                        Text(verbatim: block.reviewMessage)
                    } icon: {
                        Image(systemName: "exclamationmark.circle.fill")
                            .foregroundStyle(.orange)
                    }
                }
            }
            Section {
                if block.needsReview {
                    // What was read may be the problem, so it can be corrected here.
                    TextField(text: $sourceDraft, axis: .vertical) {
                        Text("Original", tableName: "Inspector")
                    }
                    .labelsHidden()
                    .multilineTextAlignment(.leading)
                    .focused($focus, equals: .source)
                    .onSubmit(commitText)
                } else {
                    Text(verbatim: block.sourceText)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            } header: {
                Text("Original", tableName: "Inspector")
            }
            Section {
                TextEditor(text: $draft)
                    .font(.body)
                    .frame(minHeight: 88)
                    .scrollContentBackground(.hidden)
                    .focused($focus, equals: .translation)
                    .accessibilityLabel(Text("Translation", tableName: "Inspector"))
                    // ⌘↩ commits without leaving the field.
                    .background {
                        Button("Apply", action: commitText)
                            .keyboardShortcut(.return, modifiers: .command)
                            .opacity(0)
                            .accessibilityHidden(true)
                    }
            } header: {
                Text("Translation", tableName: "Inspector")
            }
            Section("Text") {
                Picker("Font", selection: edit(\.fontFamily) { BlockEdit(index: block.index, fontFamily: $0) }) {
                    fontOptions
                }
                LabeledContent("Size") {
                    HStack(spacing: 6) {
                        TextField("Size", value: fontSize, format: .number.precision(.fractionLength(0...1)))
                            .labelsHidden()
                            .multilineTextAlignment(.trailing)
                            .frame(width: 56)
                        Text(verbatim: "px")
                            .foregroundStyle(.secondary)
                        Stepper("Size", value: fontSize, in: 4...400, step: 1)
                            .labelsHidden()
                    }
                }
                LabeledContent("Style") {
                    HStack(spacing: 8) {
                        Toggle("Bold", systemImage: "bold", isOn: edit(\.bold) { BlockEdit(index: block.index, bold: $0) })
                        Toggle("Italic", systemImage: "italic", isOn: edit(\.italic) { BlockEdit(index: block.index, italic: $0) })
                    }
                    .toggleStyle(.button)
                    .labelStyle(.iconOnly)
                }
                Picker("Alignment", selection: edit(\.alignment) { BlockEdit(index: block.index, alignment: $0) }) {
                    Label("Left", systemImage: "text.alignleft").tag(BlockAlignment.left)
                    Label("Center", systemImage: "text.aligncenter").tag(BlockAlignment.center)
                    Label("Right", systemImage: "text.alignright").tag(BlockAlignment.right)
                }
                .pickerStyle(.segmented)
                .labelStyle(.iconOnly)
                ColorPicker("Color", selection: $color, supportsOpacity: false)
            }
            Section {
                Button("Auto-Fit", systemImage: "arrow.up.left.and.arrow.down.right") {
                    commitText()
                    model.apply(BlockEdit(index: block.index), autofit: true, to: page)
                }
                .help("Fit the text to its bubble again")
                if block.needsReview, model.canKeepOriginal {
                    Button("Keep Original", systemImage: "character.bubble") {
                        model.keepOriginal(block.index, on: page)
                    }
                    .help("Leave this text untranslated")
                }
                Button("Delete", systemImage: "trash", role: .destructive) {
                    model.deleteBlock(block.index, on: page)
                }
            }
        }
        .formStyle(.grouped)
        // Leaving both text fields commits them together; moving between the two does not.
        .onChange(of: focus) {
            if focus == nil { commitText() }
        }
        .onChange(of: block.translation) {
            // Undo, auto-fit or the engine changed it under us.
            if focus == nil { draft = block.translation }
        }
        .onChange(of: block.sourceText) {
            if focus == nil { sourceDraft = block.sourceText }
        }
        .onChange(of: block.color) {
            if RGB(color) != block.color { color = Color(block.color) }
        }
        // The colour panel fires continuously while dragging; send once it settles.
        .task(id: color) {
            let picked = RGB(color)
            guard picked != block.color else { return }
            try? await Task.sleep(for: .milliseconds(250))
            guard !Task.isCancelled else { return }
            model.apply(BlockEdit(index: block.index, color: picked), to: page)
        }
        .onDisappear(perform: commitText)
    }

    @ViewBuilder private var fontOptions: some View {
        let fonts = model.fonts?.fonts ?? []
        if !fonts.contains(where: { $0.family == block.fontFamily }) {
            Text(verbatim: block.fontFamily).tag(block.fontFamily)
        }
        let custom = fonts.filter { $0.source == "custom" }
        if !custom.isEmpty {
            Section("Added") {
                ForEach(custom) { Text(verbatim: $0.display).tag($0.family) }
            }
        }
        Section("System") {
            ForEach(fonts.filter { $0.source != "custom" }) { Text(verbatim: $0.display).tag($0.family) }
        }
    }

    private var fontSize: Binding<Double> {
        edit(\.fontSize) { BlockEdit(index: block.index, fontSize: min(max($0, 4), 400)) }
    }

    private func edit<Value>(_ keyPath: KeyPath<Block, Value>, _ make: @escaping (Value) -> BlockEdit) -> Binding<Value> {
        Binding {
            block[keyPath: keyPath]
        } set: { value in
            model.apply(make(value), to: page)
        }
    }

    /// Sends whichever of the two texts changed as one edit. A withheld block is laid out
    /// again by the engine, which is what decides whether it can be drawn now.
    private func commitText() {
        let edit = BlockEdit(index: block.index,
                             translation: draft == block.translation ? nil : draft,
                             sourceText: sourceDraft == block.sourceText ? nil : sourceDraft)
        guard edit.translation != nil || edit.sourceText != nil else { return }
        model.apply(edit, autofit: block.isWithheld, to: page)
    }
}

extension Color {
    init(_ rgb: RGB) {
        self.init(.sRGB, red: Double(rgb.r) / 255, green: Double(rgb.g) / 255, blue: Double(rgb.b) / 255)
    }
}

extension RGB {
    init(_ color: Color) {
        let resolved = color.resolve(in: EnvironmentValues())
        // `Color.Resolved` is extended sRGB; clamp to the 0–255 the engine stores.
        func component(_ value: Float) -> Int { Int((min(max(value, 0), 1) * 255).rounded()) }
        self.init(r: component(resolved.red), g: component(resolved.green), b: component(resolved.blue))
    }
}
