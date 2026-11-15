import Foundation
import Testing
@testable import EngineKit

/// A tiny engine in /bin/sh: answers by method name, echoing the request id.
/// - `hello`: writes to stderr, emits an event, then answers.
/// - `first`: answer is held back until `second` arrives, so replies come out of order.
/// - `boom`: error response. `noise`: a non-JSON stdout line, then the answer. `die`: exits with 3.
private let fakeEngine = #"""
held=""
while IFS= read -r line; do
  id=$(printf '%s' "$line" | sed -n 's/.*"id":\([0-9]*\).*/\1/p')
  case "$line" in
    *'"hello"'*)
      echo "loading models" >&2
      echo '{"event":"stage_progress","stage":"ocr","done":3,"total":12}'
      echo "{\"id\":$id,\"result\":{\"protocol\":1,\"engine_version\":\"1.5.21\",\"device\":\"mps\",\"torch_installed\":true,\"models_ready\":true}}" ;;
    *'"first"'*) held=$id ;;
    *'"second"'*)
      echo "{\"id\":$id,\"result\":{\"name\":\"second\"}}"
      echo "{\"id\":$held,\"result\":{\"name\":\"first\"}}" ;;
    *'"boom"'*)
      echo "{\"id\":$id,\"error\":{\"code\":\"translator_not_ready\",\"message\":\"Needs a key.\",\"data\":{\"missing\":\"api_key\"}}}" ;;
    *'"noise"'*)
      echo "tqdm leaked here"
      echo "{\"id\":$id,\"result\":{\"ok\":true}}" ;;
    *'"die"'*) exit 3 ;;
  esac
done
"""#

private func makeClient() -> EngineClient {
    EngineClient { EngineLaunch(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", fakeEngine]) }
}

private func next(_ events: AsyncStream<EngineEvent>) async -> EngineEvent? {
    var iterator = events.makeAsyncIterator()
    return await iterator.next()
}

@Suite(.timeLimit(.minutes(1))) struct EngineClientTests {
    @Test func typedCallEventAndStderr() async throws {
        let client = makeClient()
        try await client.start()
        let hello = try await client.hello()
        #expect(hello.protocolVersion == 1 && hello.engineVersion == "1.5.21" && hello.device == "mps")
        #expect(await next(client.events) == .stageProgress(stage: "ocr", done: 3, total: 12))
        // stderr is read on its own task; give it a moment.
        for _ in 0..<100 where !(await client.diagnostics().contains("loading models")) {
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(await client.diagnostics().contains("loading models"))
        await client.stop()
    }

    @Test func responsesAreMatchedByID() async throws {
        let client = makeClient()
        try await client.start()
        let first = Task { try await client.call("first", params: nil) }
        try await Task.sleep(for: .milliseconds(200))
        let second = try await client.call("second", params: nil)
        #expect(second["name"] == "second")
        #expect(try await first.value["name"] == "first")
        await client.stop()
    }

    @Test func errorResponseBecomesEngineError() async throws {
        let client = makeClient()
        try await client.start()
        await #expect(throws: EngineError.remote(code: "translator_not_ready", message: "Needs a key.", data: ["missing": "api_key"])) {
            try await client.call("boom", params: nil)
        }
        await client.stop()
    }

    @Test func strayStdoutLineGoesToTheLog() async throws {
        let client = makeClient()
        try await client.start()
        #expect(try await client.call("noise", params: nil) == ["ok": true])
        #expect(await client.diagnostics().contains("stdout: tqdm leaked here"))
        await client.stop()
    }

    @Test func processExitFailsPendingAndReports() async throws {
        let client = makeClient()
        try await client.start()
        await #expect(throws: EngineError.terminated(status: 3)) {
            try await client.call("die", params: nil)
        }
        #expect(await next(client.events) == .terminated(status: 3))
        await #expect(throws: EngineError.notRunning) {
            try await client.call("hello", params: nil)
        }
        // Restart brings it back on the same event stream.
        try await client.start()
        #expect(try await client.hello().protocolVersion == 1)
        await client.stop()
    }

    @Test func stopDoesNotReportTermination() async throws {
        let client = makeClient()
        try await client.start()
        _ = try await client.hello()
        _ = await next(client.events)
        await client.stop()
        await #expect(throws: EngineError.notRunning) {
            try await client.call("hello", params: nil)
        }
        try await client.start()
        _ = try await client.hello()
        // The only event after the restart is the new hello's; no `.terminated` from the old process.
        #expect(await next(client.events) == .stageProgress(stage: "ocr", done: 3, total: 12))
        await client.stop()
    }

    @Test func launchFailureIsReported() async {
        let client = EngineClient { EngineLaunch(executable: URL(fileURLWithPath: "/nonexistent/python"), arguments: []) }
        await #expect(throws: EngineError.self) {
            try await client.start()
        }
    }
}

@Suite struct EngineLocatorTests {
    private func makeRepo(named name: String, in base: URL, venv: Bool = true) throws -> URL {
        let root = base.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("ballontranslator"), withIntermediateDirectories: true)
        if venv {
            let bin = root.appendingPathComponent(".venv/bin")
            try FileManager.default.createDirectory(at: bin, withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: bin.appendingPathComponent("python"),
                                                       withDestinationURL: URL(fileURLWithPath: "/bin/sh"))
        }
        return root
    }

    @Test func resolutionOrder() throws {
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("locator-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let fromSettings = try makeRepo(named: "settings", in: base)
        let fromEnvironment = try makeRepo(named: "environment", in: base)
        let containing = try makeRepo(named: "containing", in: base)
        let executable = containing.appendingPathComponent("macos-app/.build/debug/MangaTranslator")

        let walked = try EngineLocator.resolve(settings: [:], environment: [:], executable: executable)
        #expect(walked.root.path == containing.path)
        #expect(walked.python.path == containing.appendingPathComponent(".venv/bin/python").path)

        let environment = [EngineLocator.rootEnvironmentKey: fromEnvironment.path, EngineLocator.pythonEnvironmentKey: "/bin/sh"]
        let viaEnvironment = try EngineLocator.resolve(settings: [:], environment: environment, executable: executable)
        #expect(viaEnvironment.root.path == fromEnvironment.path)
        #expect(viaEnvironment.python.path == "/bin/sh")

        let viaSettings = try EngineLocator.resolve(
            settings: [EngineLocator.rootDefaultsKey: fromSettings.path], environment: environment, executable: executable)
        #expect(viaSettings.root.path == fromSettings.path)
        #expect(viaSettings.python.path == "/bin/sh")

        let launch = EngineLaunch(location: walked)
        #expect(launch.arguments == ["-m", "ballontranslator.engine"])
        #expect(launch.directory == walked.root)
        #expect(launch.environment["BALLOONTRANS_APP_ROOT"] == walked.root.path)
        #expect(launch.environment["QT_QPA_PLATFORM"] == "offscreen")
    }

    @Test func walksPastACheckoutWithoutVenv() throws {
        // A git worktree has the sources but not the venv; the outer checkout has both.
        let base = FileManager.default.temporaryDirectory.appendingPathComponent("locator-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: base) }
        let outer = try makeRepo(named: "outer", in: base)
        let worktree = try makeRepo(named: "outer/worktrees/w", in: base, venv: false)
        let found = try EngineLocator.resolve(settings: [:], environment: [:],
                                              executable: worktree.appendingPathComponent("macos-app/bin/app"))
        #expect(found.root.path == outer.path)
    }

    @Test func missingEngineThrows() {
        #expect(throws: EngineError.self) {
            try EngineLocator.resolve(settings: [:], environment: [:], executable: URL(fileURLWithPath: "/usr/bin/true"))
        }
        #expect(throws: EngineError.self) {
            try EngineLocator.resolve(settings: [EngineLocator.rootDefaultsKey: "/usr"], environment: [:], executable: nil)
        }
    }
}
