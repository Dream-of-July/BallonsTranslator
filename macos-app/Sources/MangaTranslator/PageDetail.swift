import AppCore
import EngineKit
import SwiftUI

/// The detail column: banners on top, the page, and the floating controls over it.
struct PageDetail: View {
    @Environment(AppModel.self) private var model
    @State private var showsLog = false

    var body: some View {
        ZStack {
            if let page = model.currentPage {
                PageCanvas(page: page)
                    .id(page.name)
            } else {
                ContentUnavailableView("No Page Selected", systemImage: "photo")
            }
        }
        // An inset, not an overlay: the page fits above the controls and scrolls under them when zoomed.
        .safeAreaInset(edge: .bottom, spacing: 0) {
            VStack(spacing: 10) {
                if let run = model.run {
                    RunProgress(run: run)
                }
                if let page = model.currentPage {
                    CanvasBar(page: page)
                }
            }
            .padding(.bottom, 14)
        }
        .safeAreaInset(edge: .top, spacing: 0) {
            banner
        }
        .background(Color(nsColor: .underPageBackgroundColor))
        .sheet(isPresented: $showsLog) {
            DiagnosticsView()
        }
    }

    @ViewBuilder private var banner: some View {
        if case .failed(let message) = model.engineState {
            Banner {
                EngineFailure(message: message, showsLog: $showsLog)
                    .frame(maxWidth: .infinity)
            }
        } else if let notice = model.notice {
            Banner {
                NoticeRow(notice: notice)
            }
        }
    }
}

private struct Banner<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        VStack(spacing: 0) {
            content
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
            Divider()
        }
        .background(.bar)
    }
}

private struct NoticeRow: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openSettings) private var openSettings
    var notice: AppModel.Notice

    var body: some View {
        HStack(spacing: 10) {
            Label(notice.message, systemImage: "exclamationmark.triangle.fill")
                .symbolRenderingMode(.multicolor)
                .lineLimit(2)
            Spacer(minLength: 12)
            if notice.action == .openTranslationSettings {
                Button("Open Settings") {
                    UserDefaults.standard.set(SettingsView.Pane.translation.rawValue, forKey: SettingsView.paneKey)
                    openSettings()
                }
            }
            Button("Dismiss", systemImage: "xmark") { model.notice = nil }
                .labelStyle(.iconOnly)
                .buttonStyle(.borderless)
        }
    }
}

/// Progress of the running translation, floating above the page controls.
private struct RunProgress: View {
    var run: AppModel.RunState

    var body: some View {
        HStack(spacing: 10) {
            if let fraction {
                ProgressView(value: fraction)
                    .frame(width: 120)
            } else {
                ProgressView()
                    .controlSize(.small)
            }
            Text(label)
                .monospacedDigit()
                .lineLimit(1)
        }
        .font(.callout)
        .padding(.horizontal, 16)
        .padding(.vertical, 9)
        .glassEffect()
        .accessibilityElement(children: .combine)
    }

    private var fraction: Double? {
        if run.cancelling { return nil }
        if let prepare = run.prepare {
            guard let done = prepare.done, let total = prepare.total, total > 0 else { return nil }
            return min(done / total, 1)
        }
        guard run.stage != nil, run.total > 0 else { return nil }
        return Double(run.done) / Double(run.total)
    }

    private var label: String {
        if run.cancelling { return String(localized: "Stopping…") }
        if let prepare = run.prepare {
            guard let done = prepare.done, let total = prepare.total, total > 0 else { return prepare.message }
            let bytes = ByteCountFormatStyle(style: .file)
            return "\(prepare.message)  \(Int64(done).formatted(bytes)) / \(Int64(total).formatted(bytes))"
        }
        let stage = switch run.stage {
        case "detect": String(localized: "Finding text")
        case "ocr": String(localized: "Reading text")
        case "translate": String(localized: "Translating")
        case "inpaint": String(localized: "Cleaning pages")
        default: String(localized: "Getting ready…")
        }
        return run.stage == nil ? stage : "\(stage)  \(run.done) / \(run.total)"
    }
}

/// Floating controls under the page: original/translated and zoom.
private struct CanvasBar: View {
    @Environment(AppModel.self) private var model
    var page: Page

    var body: some View {
        @Bindable var model = model
        GlassEffectContainer(spacing: 10) {
            HStack(spacing: 10) {
                if model.renderingPages.contains(page.name) {
                    ProgressView()
                        .controlSize(.small)
                        .padding(8)
                        .glassEffect()
                        .accessibilityLabel("Updating page")
                }
                if page.result != nil {
                    Toggle("Original", systemImage: "photo", isOn: $model.showsOriginal)
                        .toggleStyle(.button)
                        .labelStyle(.titleAndIcon)
                        .help("Show the untranslated page")
                }
                Button("Zoom Out", systemImage: "minus.magnifyingglass") { model.zoomOut() }
                    .disabled(model.zoom <= AppModel.zoomRange.lowerBound)
                Button("Zoom to Fit", systemImage: "arrow.up.left.and.down.right.magnifyingglass") { model.zoomToFit() }
                    .disabled(model.zoom == 1)
                Button("Zoom In", systemImage: "plus.magnifyingglass") { model.zoomIn() }
                    .disabled(model.zoom >= AppModel.zoomRange.upperBound)
            }
            .labelStyle(.iconOnly)
            .buttonStyle(.glass)
            .controlSize(.large)
        }
    }
}

/// One page, zoomable and scrollable, with its text blocks as editable outlines.
struct PageCanvas: View {
    @Environment(AppModel.self) private var model
    var page: Page
    @State private var image: CGImage?
    @State private var zoomAtPinchStart: Double?
    @FocusState private var isFocused: Bool

    private static let padding: CGFloat = 24

    private var imagePath: String {
        model.showsOriginal ? page.source : page.result ?? page.source
    }

    var body: some View {
        GeometryReader { proxy in
            let fit = PageGeometry.fitScale(image: page.size, container: proxy.size, padding: Self.padding)
            let geometry = PageGeometry(imageSize: page.size, scale: fit * model.zoom)
            ScrollView([.horizontal, .vertical]) {
                ZStack(alignment: .topLeading) {
                    pageImage
                        .frame(width: geometry.viewSize.width, height: geometry.viewSize.height)
                        .contentShape(.rect)
                        .onTapGesture {
                            model.selectedBlock = nil
                            isFocused = true
                        }
                    if page.result != nil, !model.showsOriginal {
                        BlockLayer(page: page, geometry: geometry) { isFocused = true }
                            .allowsHitTesting(model.canEdit(page: page.name))
                    }
                }
                .coordinateSpace(.named(BlockLayer.space))
                .padding(Self.padding)
                .frame(minWidth: proxy.size.width, minHeight: proxy.size.height)
            }
        }
        .focusable()
        .focusEffectDisabled()
        .focused($isFocused)
        .onKeyPress(keys: [.leftArrow, .rightArrow, .upArrow, .downArrow]) { press in
            nudge(press) ? .handled : .ignored
        }
        .onKeyPress(.escape) {
            guard model.selectedBlock != nil else { return .ignored }
            model.selectedBlock = nil
            return .handled
        }
        .onDeleteCommand {
            if let block = model.selectedBlock {
                model.deleteBlock(block, on: page.name)
            }
        }
        .gesture(
            MagnifyGesture()
                .onChanged { value in
                    let start = zoomAtPinchStart ?? model.zoom
                    zoomAtPinchStart = start
                    model.zoom = min(max(start * value.magnification, AppModel.zoomRange.lowerBound), AppModel.zoomRange.upperBound)
                }
                .onEnded { _ in zoomAtPinchStart = nil }
        )
        // Keeps the previous image on screen until the new one is decoded, so edits don't flash.
        .task(id: "\(imagePath)#\(page.revision)") {
            if let loaded = await ImageLoader.load(imagePath) {
                image = loaded
            }
        }
    }

    @ViewBuilder private var pageImage: some View {
        if let image {
            Image(decorative: image, scale: 1)
                .resizable()
                .interpolation(.high)
                .accessibilityLabel(Text(verbatim: page.name))
        } else {
            Rectangle().fill(.quaternary)
        }
    }

    private func nudge(_ press: KeyPress) -> Bool {
        guard let block = model.currentBlock else { return false }
        let step: CGFloat = press.modifiers.contains(.shift) ? 10 : 1
        let delta: CGSize = switch press.key {
        case .leftArrow: CGSize(width: -step, height: 0)
        case .rightArrow: CGSize(width: step, height: 0)
        case .upArrow: CGSize(width: 0, height: -step)
        default: CGSize(width: 0, height: step)
        }
        model.apply(BlockEdit(index: block.index, rect: BlockGeometry.moved(block.rect, by: delta, within: page.size)),
                    to: page.name)
        return true
    }
}

/// Selectable, movable, resizable outlines of a page's text blocks. Only the outline
/// follows the pointer; the text is redrawn by the engine after the gesture ends.
struct BlockLayer: View {
    static let space = "page"

    @Environment(AppModel.self) private var model
    var page: Page
    var geometry: PageGeometry
    var takeFocus: () -> Void
    @State private var dragged: (index: Int, rect: CGRect)?
    @State private var hovered: Int?

    var body: some View {
        ZStack(alignment: .topLeading) {
            ForEach(page.blocks) { block in
                let frame = geometry.toView(rect(of: block))
                let isSelected = model.selectedBlock == block.index
                Rectangle()
                    .strokeBorder(Color.accentColor.opacity(isSelected ? 1 : hovered == block.index ? 0.55 : 0),
                                  lineWidth: isSelected ? 2 : 1)
                    .frame(width: frame.width, height: frame.height)
                    .contentShape(.rect)
                    .onHover { inside in
                        if inside { hovered = block.index } else if hovered == block.index { hovered = nil }
                    }
                    .onTapGesture { select(block) }
                    .gesture(move(block))
                    .rotationEffect(.degrees(block.angle))
                    .position(x: frame.midX, y: frame.midY)
                    .accessibilityElement()
                    .accessibilityLabel(Text(verbatim: block.translation.isEmpty ? block.sourceText : block.translation))
                    .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
                    .accessibilityAction { select(block) }
            }
            if let block = page.blocks.first(where: { $0.index == model.selectedBlock }) {
                ForEach(ResizeHandle.allCases, id: \.self) { handle in
                    let point = geometry.toView(BlockGeometry.position(of: handle.unit, rect: rect(of: block), angle: block.angle))
                    Circle()
                        .fill(.white)
                        .strokeBorder(Color.accentColor, lineWidth: 1.5)
                        .frame(width: 9, height: 9)
                        .frame(width: 20, height: 20)
                        .contentShape(.rect)
                        .gesture(resize(block, handle))
                        .position(point)
                        .accessibilityHidden(true)
                }
            }
        }
        .frame(width: geometry.viewSize.width, height: geometry.viewSize.height, alignment: .topLeading)
    }

    private func rect(of block: Block) -> CGRect {
        if let dragged, dragged.index == block.index { dragged.rect } else { block.rect }
    }

    private func select(_ block: Block) {
        model.selectedBlock = block.index
        takeFocus()
    }

    private func move(_ block: Block) -> some Gesture {
        DragGesture(minimumDistance: 3, coordinateSpace: .named(Self.space))
            .onChanged { value in
                select(block)
                dragged = (block.index, BlockGeometry.moved(block.rect, by: geometry.toImage(value.translation), within: page.size))
            }
            .onEnded { _ in commit() }
    }

    private func resize(_ block: Block, _ handle: ResizeHandle) -> some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .named(Self.space))
            .onChanged { value in
                dragged = (block.index, BlockGeometry.resized(block.rect, angle: block.angle, handle: handle,
                                                              by: geometry.toImage(value.translation)))
            }
            .onEnded { _ in commit() }
    }

    private func commit() {
        guard let dragged else { return }
        self.dragged = nil
        // Whole pixels: the box is stored in image pixels.
        let rect = CGRect(x: dragged.rect.minX.rounded(), y: dragged.rect.minY.rounded(),
                          width: dragged.rect.width.rounded(), height: dragged.rect.height.rounded())
        model.apply(BlockEdit(index: dragged.index, rect: rect), to: page.name)
    }
}
