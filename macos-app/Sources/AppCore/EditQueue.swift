import EngineKit

/// Serialises block edits per page: at most one request in flight for a page, and
/// everything that arrives meanwhile is merged into a single follow-up request.
///
/// >>> var queue = EditQueue()
/// >>> queue.enqueue(BlockEdit(index: 0, fontSize: 30), page: "001.png")   // Batch: send now
/// >>> queue.enqueue(BlockEdit(index: 0, fontSize: 32), page: "001.png")   // nil: waits
/// >>> queue.enqueue(BlockEdit(index: 0, bold: true), page: "001.png")     // nil: merged
/// >>> queue.finish(page: "001.png")   // Batch(fontSize: 32, bold: true)
/// >>> queue.finish(page: "001.png")   // nil: page is idle again
public struct EditQueue: Sendable {
    public struct Batch: Equatable, Sendable {
        public var page: String
        public var edits: [BlockEdit] = []
        public var autofit = false
        /// Sent as `delete_blocks` after `edits`.
        public var deletions: [Int] = []
    }

    private var inFlight: Set<String> = []
    private var waiting: [String: Batch] = [:]

    public init() {}

    /// Returns the batch to send now, or nil when the edit was queued behind a running request.
    public mutating func enqueue(_ edit: BlockEdit, autofit: Bool = false, page: String) -> Batch? {
        var batch = waiting[page] ?? Batch(page: page)
        if let position = batch.edits.firstIndex(where: { $0.index == edit.index }) {
            batch.edits[position] = batch.edits[position].merging(edit)
        } else {
            batch.edits.append(edit)
        }
        batch.autofit = batch.autofit || autofit
        return submit(batch)
    }

    public mutating func enqueueDeletion(of index: Int, page: String) -> Batch? {
        var batch = waiting[page] ?? Batch(page: page)
        batch.edits.removeAll { $0.index == index }
        if !batch.deletions.contains(index) { batch.deletions.append(index) }
        return submit(batch)
    }

    /// Call when a batch's requests have completed. Returns the next batch for the page, if any.
    public mutating func finish(page: String) -> Batch? {
        if let next = waiting.removeValue(forKey: page) {
            return next
        }
        inFlight.remove(page)
        return nil
    }

    /// Edits queued behind the running request; they are not in the engine's reply yet.
    public func waitingEdits(page: String) -> [BlockEdit] {
        waiting[page]?.edits ?? []
    }

    public func isBusy(page: String) -> Bool {
        inFlight.contains(page)
    }

    public mutating func reset() {
        inFlight.removeAll()
        waiting.removeAll()
    }

    private mutating func submit(_ batch: Batch) -> Batch? {
        if inFlight.contains(batch.page) {
            waiting[batch.page] = batch
            return nil
        }
        inFlight.insert(batch.page)
        return batch
    }
}
