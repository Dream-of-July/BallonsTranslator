import CoreGraphics

/// Maps between source-image pixels (origin top-left) and the page view's points.
///
/// >>> let geometry = PageGeometry(imageSize: CGSize(width: 800, height: 1200), scale: 0.5)
/// >>> geometry.toView(CGPoint(x: 100, y: 40))   // (50, 20)
/// >>> geometry.toImage(CGPoint(x: 50, y: 20))   // (100, 40)
public struct PageGeometry: Equatable, Sendable {
    public var imageSize: CGSize
    /// View points per image pixel.
    public var scale: CGFloat

    public init(imageSize: CGSize, scale: CGFloat) {
        self.imageSize = imageSize
        self.scale = scale
    }

    public var viewSize: CGSize {
        CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }

    /// Scale at which the whole image fits `container` with `padding` on every side.
    public static func fitScale(image: CGSize, container: CGSize, padding: CGFloat = 0) -> CGFloat {
        guard image.width > 0, image.height > 0 else { return 1 }
        let width = max(container.width - 2 * padding, 1)
        let height = max(container.height - 2 * padding, 1)
        return min(width / image.width, height / image.height)
    }

    public func toView(_ point: CGPoint) -> CGPoint {
        CGPoint(x: point.x * scale, y: point.y * scale)
    }

    public func toImage(_ point: CGPoint) -> CGPoint {
        CGPoint(x: point.x / scale, y: point.y / scale)
    }

    public func toView(_ rect: CGRect) -> CGRect {
        CGRect(x: rect.minX * scale, y: rect.minY * scale, width: rect.width * scale, height: rect.height * scale)
    }

    public func toImage(_ size: CGSize) -> CGSize {
        CGSize(width: size.width / scale, height: size.height / scale)
    }
}

/// The eight resize handles of a block; `unit` is the handle's position in the box, -1…1 per axis.
public enum ResizeHandle: CaseIterable, Sendable {
    case topLeft, top, topRight, right, bottomRight, bottom, bottomLeft, left

    public var unit: CGPoint {
        switch self {
        case .topLeft: CGPoint(x: -1, y: -1)
        case .top: CGPoint(x: 0, y: -1)
        case .topRight: CGPoint(x: 1, y: -1)
        case .right: CGPoint(x: 1, y: 0)
        case .bottomRight: CGPoint(x: 1, y: 1)
        case .bottom: CGPoint(x: 0, y: 1)
        case .bottomLeft: CGPoint(x: -1, y: 1)
        case .left: CGPoint(x: -1, y: 0)
        }
    }
}

/// Geometry of a block: an unrotated box in image pixels, shown rotated by `angle`
/// degrees (clockwise, y down) around its centre.
public enum BlockGeometry {
    public static let minimumSize: CGFloat = 12

    static func rotate(_ vector: CGPoint, degrees: Double) -> CGPoint {
        let radians = degrees * .pi / 180
        let (sine, cosine) = (sin(radians), cos(radians))
        return CGPoint(x: vector.x * cosine - vector.y * sine, y: vector.x * sine + vector.y * cosine)
    }

    /// Whether `point` (image pixels) is inside the rotated box.
    public static func contains(_ point: CGPoint, rect: CGRect, angle: Double) -> Bool {
        let local = rotate(CGPoint(x: point.x - rect.midX, y: point.y - rect.midY), degrees: -angle)
        return abs(local.x) <= rect.width / 2 && abs(local.y) <= rect.height / 2
    }

    /// Where a point of the box (`unit`: -1…1 per axis) ends up in the image after rotation.
    public static func position(of unit: CGPoint, rect: CGRect, angle: Double) -> CGPoint {
        let offset = rotate(CGPoint(x: unit.x * rect.width / 2, y: unit.y * rect.height / 2), degrees: angle)
        return CGPoint(x: rect.midX + offset.x, y: rect.midY + offset.y)
    }

    /// Moves the box, keeping its centre inside the image.
    public static func moved(_ rect: CGRect, by delta: CGSize, within imageSize: CGSize) -> CGRect {
        let centerX = min(max(rect.midX + delta.width, 0), imageSize.width)
        let centerY = min(max(rect.midY + delta.height, 0), imageSize.height)
        return CGRect(x: centerX - rect.width / 2, y: centerY - rect.height / 2, width: rect.width, height: rect.height)
    }

    /// Drags `handle` by `delta` (image pixels, image axes). The opposite edge or corner
    /// stays where it is on the page, also for rotated boxes.
    public static func resized(_ rect: CGRect, angle: Double, handle: ResizeHandle, by delta: CGSize) -> CGRect {
        let unit = handle.unit
        let local = rotate(CGPoint(x: delta.width, y: delta.height), degrees: -angle)
        let width = max(rect.width + unit.x * local.x, minimumSize)
        let height = max(rect.height + unit.y * local.y, minimumSize)
        let anchor = position(of: CGPoint(x: -unit.x, y: -unit.y), rect: rect, angle: angle)
        // Offset from the new centre to the anchor, in page axes.
        let offset = rotate(CGPoint(x: -unit.x * width / 2, y: -unit.y * height / 2), degrees: angle)
        let center = CGPoint(x: anchor.x - offset.x, y: anchor.y - offset.y)
        return CGRect(x: center.x - width / 2, y: center.y - height / 2, width: width, height: height)
    }
}
