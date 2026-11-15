// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "MangaTranslator",
    platforms: [.macOS(.v26)],
    targets: [
        // Protocol models, the process client and the mock engine.
        .target(name: "EngineKit"),
        // View-independent app state: project, run, edit queue, geometry.
        .target(name: "AppCore", dependencies: ["EngineKit"]),
        .executableTarget(name: "MangaTranslator", dependencies: ["AppCore", "EngineKit"]),
        .testTarget(name: "MangaTranslatorTests", dependencies: ["AppCore", "EngineKit"]),
    ]
)
