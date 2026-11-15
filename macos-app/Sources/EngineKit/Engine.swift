import Foundation

/// The engine as the app sees it. `EngineClient` talks to the Python process,
/// `MockEngine` answers in-process; both speak raw protocol JSON through `call`.
public protocol Engine: Sendable {
    /// Engine events in arrival order. One consumer; survives restarts.
    var events: AsyncStream<EngineEvent> { get }
    func start() async throws
    /// Ends the engine without emitting `.terminated`.
    func stop() async
    /// Sends one request and returns its `result`, or throws `EngineError`.
    func call(_ method: String, params: JSONValue?) async throws -> JSONValue
    /// Recent stderr / diagnostic lines, oldest first.
    func diagnostics() async -> [String]
}

// Typed wrappers for every method in PROTOCOL.md.
extension Engine {
    private func request<T: Decodable>(_ method: String, _ params: JSONValue? = nil) async throws -> T {
        let value = try await call(method, params: params)
        do {
            return try value.decode()
        } catch {
            throw EngineError.invalidResponse("\(method): \(error)")
        }
    }

    public func hello() async throws -> Hello {
        try await request("hello")
    }

    public func openProject(path: String) async throws -> Project {
        try await request("open_project", ["path": .string(path)])
    }

    public func page(named name: String) async throws -> Page {
        try await request("get_page", ["name": .string(name)])
    }

    public func run(pages: [String]? = nil, mode: RunMode) async throws {
        var params: [String: JSONValue] = ["mode": .string(mode.rawValue)]
        if let pages { params["pages"] = .array(pages.map(JSONValue.string)) }
        _ = try await call("run", params: .object(params))
    }

    public func cancel() async throws {
        _ = try await call("cancel", params: nil)
    }

    public func updateBlocks(page: String, edits: [BlockEdit], autofit: Bool = false) async throws -> Page {
        var params: [String: JSONValue] = ["page": .string(page), "edits": try JSONValue(encoding: edits)]
        if autofit { params["autofit"] = true }
        return try await request("update_blocks", .object(params))
    }

    public func deleteBlocks(page: String, indexes: [Int]) async throws -> Page {
        try await request("delete_blocks", ["page": .string(page), "indexes": .array(indexes.map { .number(Double($0)) })])
    }

    public func listFonts() async throws -> FontList {
        try await request("list_fonts")
    }

    public func settings() async throws -> EngineSettings {
        try await request("get_settings")
    }

    public func setSettings(_ patch: SettingsPatch) async throws -> EngineSettings {
        try await request("set_settings", try JSONValue(encoding: patch))
    }

    public func createProfile(template: String) async throws -> Profile {
        try await request("create_profile", ["template": .string(template)])
    }

    public func updateProfile(id: String, _ patch: ProfilePatch) async throws -> Profile {
        guard case .object(var params) = try JSONValue(encoding: patch) else {
            throw EngineError.invalidResponse("update_profile")
        }
        params["id"] = .string(id)
        return try await request("update_profile", .object(params))
    }

    public func deleteProfile(id: String) async throws -> EngineSettings {
        try await request("delete_profile", ["id": .string(id)])
    }

    public func testProfile(id: String) async throws -> ProfileTestResult {
        try await request("test_profile", ["id": .string(id)])
    }

    public func export(to destination: String, what: ExportKind) async throws -> [String] {
        let result: ExportResult = try await request(
            "export", ["destination": .string(destination), "what": .string(what.rawValue)])
        return result.files
    }
}
