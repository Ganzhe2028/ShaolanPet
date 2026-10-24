import AppKit
import CoreImage
import Foundation
import Vision

// 从一张图里抠出主体，输出透明背景 PNG。
// 桌宠的头像素材就靠这个，以后做任何角色都能用。
//
// 用法: cutout <输入图> <输出图> [最大宽度]
//   - 前景有多个实例（人 + 画面上的文字块）时，只保留面积最大的那个
//   - 输出会自动居中补成正方形，桌宠窗口是方的，省得再处理

let args = CommandLine.arguments
guard args.count >= 3 else {
    print("usage: cutout <input> <output> [maxWidth]")
    exit(1)
}
let inURL = URL(fileURLWithPath: args[1])
let outURL = URL(fileURLWithPath: args[2])
let maxWidth = args.count >= 4 ? (Double(args[3]) ?? 1200) : 1200

guard let src = NSImage(contentsOf: inURL),
      let cg = src.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("读取图片失败: \(inURL.path)")
    exit(1)
}
print("input: \(cg.width)x\(cg.height)")

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
let request = VNGenerateForegroundInstanceMaskRequest()
do {
    try handler.perform([request])
} catch {
    print("Vision 抠图失败: \(error)")
    exit(1)
}
guard let obs = request.results?.first, !obs.allInstances.isEmpty else {
    print("没有找到前景主体")
    exit(1)
}
print("instances: \(obs.allInstances.count)")

// 多个实例时挑面积最大的，避免把字幕块也抠进来
let buffer: CVPixelBuffer
if obs.allInstances.count == 1 {
    buffer = try obs.generateMaskedImage(ofInstances: obs.allInstances, from: handler, croppedToInstancesExtent: true)
} else {
    var best: (index: Int, area: Int) = (obs.allInstances.first!, 0)
    for index in obs.allInstances {
        let mask = try obs.generateScaledMaskForImage(forInstances: IndexSet(integer: index), from: handler)
        let w = CVPixelBufferGetWidth(mask), h = CVPixelBufferGetHeight(mask)
        CVPixelBufferLockBaseAddress(mask, .readOnly)
        let base = CVPixelBufferGetBaseAddress(mask)!.assumingMemoryBound(to: Float32.self)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(mask) / MemoryLayout<Float32>.size
        var sum: Float = 0
        for y in stride(from: 0, to: h, by: 4) {
            for x in stride(from: 0, to: w, by: 4) { sum += base[y * bytesPerRow + x] }
        }
        CVPixelBufferUnlockBaseAddress(mask, .readOnly)
        let area = Int(sum / 16)
        if area > best.area { best = (index, area) }
    }
    print("picked instance \(best.0) area≈\(best.1)")
    buffer = try obs.generateMaskedImage(ofInstances: IndexSet(integer: best.0), from: handler, croppedToInstancesExtent: true)
}

let ci = CIImage(cvPixelBuffer: buffer)
let ctx = CIContext()
guard let featCG = ctx.createCGImage(ci, from: ci.extent) else {
    print("渲染失败")
    exit(1)
}
print("cutout: \(featCG.width)x\(featCG.height)")

// 缩放到目标宽度
let scale = min(1.0, maxWidth / Double(featCG.width))
let outW = Int(Double(featCG.width) * scale)
let outH = Int(Double(featCG.height) * scale)
let side = max(outW, outH)
guard let canvas = CGContext(
    data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: 0,
    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
) else {
    print("创建画布失败")
    exit(1)
}
canvas.interpolationQuality = .high
canvas.draw(featCG, in: CGRect(x: (side - outW) / 2, y: (side - outH) / 2, width: outW, height: outH))
guard let finalCG = canvas.makeImage() else {
    print("合成失败")
    exit(1)
}

let rep = NSBitmapImageRep(cgImage: finalCG)
guard let data = rep.representation(using: .png, properties: [:]) else {
    print("编码 PNG 失败")
    exit(1)
}
try data.write(to: outURL)
print("wrote \(outURL.path) \(side)x\(side) \(data.count / 1024)KB")
