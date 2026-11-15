import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum ProjectImporter {
    /// `~/Library/Application Support/MangaTranslator/Projects`
    public static var projectsDirectory: URL {
        URL.applicationSupportDirectory.appendingPathComponent("MangaTranslator/Projects", isDirectory: true)
    }

    public static func isImage(_ url: URL) -> Bool {
        UTType(filenameExtension: url.pathExtension)?.conforms(to: .image) ?? false
    }

    public static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
    }

    /// Copies loose image files into a new project folder under `base` and returns it.
    /// The folder is named after the images' parent folder, with a number added if taken.
    @concurrent
    public static func importImages(_ urls: [URL], into base: URL = projectsDirectory) async throws -> URL {
        let images = urls.filter(isImage)
        guard let first = images.first else {
            throw CocoaError(.fileReadUnsupportedScheme)
        }
        let fileManager = FileManager.default
        try fileManager.createDirectory(at: base, withIntermediateDirectories: true)
        let name = first.deletingLastPathComponent().lastPathComponent
        var folder = base.appendingPathComponent(name, isDirectory: true)
        var suffix = 2
        while fileManager.fileExists(atPath: folder.path) {
            folder = base.appendingPathComponent("\(name) \(suffix)", isDirectory: true)
            suffix += 1
        }
        try fileManager.createDirectory(at: folder, withIntermediateDirectories: false)
        for image in images {
            try fileManager.copyItem(at: image, to: folder.appendingPathComponent(image.lastPathComponent))
        }
        return folder
    }
}

public enum ImageLoader {
    /// Decodes an image off the main actor. `maxPixelSize` produces a thumbnail.
    @concurrent
    public static func load(_ path: String, maxPixelSize: Int? = nil) async -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
        if let maxPixelSize {
            let options: [CFString: Any] = [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceCreateThumbnailWithTransform: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixelSize,
            ]
            return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        }
        return CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary)
    }
}
