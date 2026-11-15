import Foundation

/// Where the Python engine lives.
public struct EngineLocation: Equatable, Sendable {
    /// Repo root: contains `ballontranslator/`.
    public var root: URL
    public var python: URL

    public init(root: URL, python: URL) {
        self.root = root
        self.python = python
    }
}

public enum EngineLocator {
    public static let rootDefaultsKey = "engineRoot"
    public static let pythonDefaultsKey = "enginePython"
    public static let rootEnvironmentKey = "MANGA_ENGINE_ROOT"
    public static let pythonEnvironmentKey = "MANGA_ENGINE_PYTHON"

    /// Resolution order: explicit setting → environment → the repo containing the running binary.
    ///
    /// - Parameters:
    ///   - settings: values of `rootDefaultsKey` / `pythonDefaultsKey` (empty strings count as unset).
    ///   - executable: where to start walking up from.
    public static func resolve(
        settings: [String: String] = storedSettings(),
        environment: [String: String] = ProcessInfo.processInfo.environment,
        executable: URL? = Bundle.main.executableURL,
        fileManager: FileManager = .default
    ) throws -> EngineLocation {
        func value(_ settingsKey: String, _ environmentKey: String) -> URL? {
            let raw = [settings[settingsKey], environment[environmentKey]]
                .compactMap { $0?.trimmingCharacters(in: .whitespaces) }
                .first { !$0.isEmpty }
            return raw.map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
        }
        func hasEngine(_ root: URL) -> Bool {
            var isDirectory: ObjCBool = false
            return fileManager.fileExists(atPath: root.appendingPathComponent("ballontranslator").path,
                                          isDirectory: &isDirectory) && isDirectory.boolValue
        }
        func defaultPython(_ root: URL) -> URL {
            // Not resolved: a venv only works when started through its own `bin/python` path.
            root.appendingPathComponent(".venv/bin/python")
        }

        let explicitPython = value(pythonDefaultsKey, pythonEnvironmentKey)
        var root = value(rootDefaultsKey, rootEnvironmentKey)
        if root == nil, var directory = executable?.deletingLastPathComponent() {
            while directory.path != "/" {
                if hasEngine(directory),
                   fileManager.isExecutableFile(atPath: (explicitPython ?? defaultPython(directory)).path) {
                    root = directory
                    break
                }
                directory.deleteLastPathComponent()
            }
        }
        guard let root else {
            throw EngineError.launchFailed(String(localized: "Can't find the translation engine. Set its location in Settings."))
        }
        guard hasEngine(root) else {
            throw EngineError.launchFailed(String(localized: "No engine in \(root.path)."))
        }
        let python = explicitPython ?? defaultPython(root)
        guard fileManager.isExecutableFile(atPath: python.path) else {
            throw EngineError.launchFailed(String(localized: "Python not found at \(python.path)."))
        }
        return EngineLocation(root: root, python: python)
    }

    public static func storedSettings(_ defaults: UserDefaults = .standard) -> [String: String] {
        var settings: [String: String] = [:]
        for key in [rootDefaultsKey, pythonDefaultsKey] {
            settings[key] = defaults.string(forKey: key)
        }
        return settings
    }
}
