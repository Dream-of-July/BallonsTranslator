import Foundation

/// How to spawn the engine process.
public struct EngineLaunch: Equatable, Sendable {
    public var executable: URL
    public var arguments: [String]
    public var directory: URL?
    /// Added on top of the app's own environment.
    public var environment: [String: String]

    public init(executable: URL, arguments: [String], directory: URL? = nil, environment: [String: String] = [:]) {
        self.executable = executable
        self.arguments = arguments
        self.directory = directory
        self.environment = environment
    }

    /// `python -m ballontranslator.engine` in `location.root`, as PROTOCOL.md requires.
    public init(location: EngineLocation) {
        self.init(
            executable: location.python,
            arguments: ["-m", "ballontranslator.engine"],
            directory: location.root,
            environment: [
                "BALLOONTRANS_APP_ROOT": location.root.path,
                "QT_QPA_PLATFORM": "offscreen",
                "PYTHONUNBUFFERED": "1",
            ]
        )
    }
}

/// Drives the engine as a child process: JSON lines on stdin/stdout, stderr kept for diagnostics.
public actor EngineClient: Engine {
    public nonisolated let events: AsyncStream<EngineEvent>
    private let eventSink: AsyncStream<EngineEvent>.Continuation
    private let launch: @Sendable () throws -> EngineLaunch

    private var process: Process?
    private var stdin: FileHandle?
    private var pending: [Int: CheckedContinuation<JSONValue, Error>] = [:]
    private var nextID = 1
    /// Bumped on every start/stop so readers of an old process cannot touch new state.
    private var generation = 0
    private var log: [String] = []
    private let logLimit = 1000

    /// `launch` is evaluated on every `start()`, so a changed engine location applies on restart.
    public init(launch: @escaping @Sendable () throws -> EngineLaunch) {
        self.launch = launch
        (events, eventSink) = AsyncStream.makeStream(of: EngineEvent.self, bufferingPolicy: .unbounded)
    }

    public func start() async throws {
        guard process == nil else { return }
        let configuration = try launch()

        let process = Process()
        process.executableURL = configuration.executable
        process.arguments = configuration.arguments
        process.currentDirectoryURL = configuration.directory
        process.environment = ProcessInfo.processInfo.environment.merging(configuration.environment) { _, new in new }
        let input = Pipe(), output = Pipe(), errors = Pipe()
        process.standardInput = input
        process.standardOutput = output
        process.standardError = errors
        do {
            try process.run()
        } catch {
            throw EngineError.launchFailed(error.localizedDescription)
        }
        // A write to a dead engine must fail with EPIPE instead of killing the app with SIGPIPE.
        _ = fcntl(input.fileHandleForWriting.fileDescriptor, F_SETNOSIGPIPE, 1)

        generation += 1
        let generation = generation
        self.process = process
        stdin = input.fileHandleForWriting
        appendLog("— engine started: \(configuration.executable.path) \(configuration.arguments.joined(separator: " "))")

        let stdout = lines(of: output.fileHandleForReading)
        Task { [weak self] in
            for await line in stdout {
                await self?.receive(line, generation: generation)
            }
            // EOF: every response the engine wrote has been delivered by now.
            await self?.outputEnded(generation: generation)
        }
        let stderr = lines(of: errors.fileHandleForReading)
        Task { [weak self] in
            for await line in stderr {
                await self?.appendLog(line)
            }
        }
    }

    public func stop() async {
        guard let process else { return }
        generation += 1
        self.process = nil
        // Closing stdin is the protocol's EOF shutdown; terminate() covers an engine that ignores it.
        try? stdin?.close()
        stdin = nil
        failPending(with: EngineError.notRunning)
        appendLog("— engine stopped")
        let box = ProcessBox(process)
        Task.detached {
            try? await Task.sleep(for: .seconds(2))
            if box.process.isRunning { box.process.terminate() }
        }
    }

    public func call(_ method: String, params: JSONValue?) async throws -> JSONValue {
        guard let stdin else { throw EngineError.notRunning }
        let id = nextID
        nextID += 1
        var request: [String: JSONValue] = ["id": .number(Double(id)), "method": .string(method)]
        if let params { request["params"] = params }
        var line = try JSONEncoder().encode(JSONValue.object(request))
        line.append(0x0A)
        do {
            try stdin.write(contentsOf: line)
        } catch {
            throw EngineError.notRunning
        }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { pending[id] = $0 }
        } onCancel: {
            Task { await self.abandon(id) }
        }
    }

    public func diagnostics() -> [String] {
        log
    }

    // MARK: - Private

    private func abandon(_ id: Int) {
        pending.removeValue(forKey: id)?.resume(throwing: CancellationError())
    }

    private func receive(_ line: String, generation: Int) {
        guard generation == self.generation else { return }
        guard let message = try? JSONDecoder().decode(JSONValue.self, from: Data(line.utf8)),
              case .object = message else {
            appendLog("stdout: \(line)")
            return
        }
        if message["event"] != nil {
            if let event = try? message.decode(EngineEvent.self) {
                eventSink.yield(event)
            } else {
                appendLog("unknown event: \(line)")
            }
            return
        }
        guard let id = message["id"]?.intValue, let continuation = pending.removeValue(forKey: id) else {
            appendLog("unmatched response: \(line)")
            return
        }
        if let error = message["error"] {
            continuation.resume(throwing: EngineError.remote(
                code: error["code"]?.stringValue ?? ErrorCode.internal.rawValue,
                message: error["message"]?.stringValue ?? "",
                data: error["data"]))
        } else {
            continuation.resume(returning: message["result"] ?? .null)
        }
    }

    private func outputEnded(generation: Int) async {
        guard generation == self.generation, let process else { return }
        let box = ProcessBox(process)
        // stdout closes a moment before the exit status is available.
        for _ in 0..<40 where box.process.isRunning {
            try? await Task.sleep(for: .milliseconds(50))
        }
        guard generation == self.generation else { return }
        let status: Int32? = box.process.isRunning ? nil : box.process.terminationStatus
        self.process = nil
        stdin = nil
        failPending(with: EngineError.terminated(status: status))
        appendLog("— engine exited" + (status.map { " with status \($0)" } ?? ""))
        eventSink.yield(.terminated(status: status))
    }

    private func failPending(with error: Error) {
        let waiting = pending
        pending.removeAll()
        for continuation in waiting.values {
            continuation.resume(throwing: error)
        }
    }

    private func appendLog(_ line: String) {
        log.append(line)
        if log.count > logLimit {
            log.removeFirst(log.count - logLimit)
        }
    }
}

/// Lines of a pipe, in order, as they arrive. (`FileHandle.bytes` holds data back until
/// its buffer fills, which stalls a request/response protocol.)
private func lines(of handle: FileHandle) -> AsyncStream<String> {
    AsyncStream { continuation in
        let buffer = LineBuffer()
        handle.readabilityHandler = { handle in
            let data = handle.availableData
            if data.isEmpty {
                handle.readabilityHandler = nil
                if let rest = buffer.flush() { continuation.yield(rest) }
                continuation.finish()
            } else {
                buffer.append(data).forEach { continuation.yield($0) }
            }
        }
    }
}

/// Only touched from one handle's readability callbacks, which are serial.
private final class LineBuffer: @unchecked Sendable {
    private var pending = Data()

    func append(_ data: Data) -> [String] {
        pending.append(data)
        var lines: [String] = []
        while let newline = pending.firstIndex(of: 0x0A) {
            lines.append(String(decoding: pending[pending.startIndex..<newline], as: UTF8.self))
            pending.removeSubrange(pending.startIndex...newline)
        }
        return lines.filter { !$0.isEmpty }
    }

    func flush() -> String? {
        defer { pending.removeAll() }
        return pending.isEmpty ? nil : String(decoding: pending, as: UTF8.self)
    }
}

/// `Process` is not Sendable; it is only polled for `isRunning` / asked to terminate across tasks.
private struct ProcessBox: @unchecked Sendable {
    let process: Process
    init(_ process: Process) { self.process = process }
}
