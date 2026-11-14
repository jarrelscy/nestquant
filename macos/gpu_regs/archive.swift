import Metal
import Foundation
// usage: archive in.metal out.bin : compile every function into a binary archive and serialize it
let dev = MTLCreateSystemDefaultDevice()!
let args = CommandLine.arguments
let src = try! String(contentsOfFile: args[1], encoding: .utf8)
let opts = MTLCompileOptions()
opts.mathMode = .safe
let lib = try! dev.makeLibrary(source: src, options: opts)
let ad = MTLBinaryArchiveDescriptor()
let archive = try! dev.makeBinaryArchive(descriptor: ad)
for name in lib.functionNames {
    let d = MTLComputePipelineDescriptor()
    d.computeFunction = lib.makeFunction(name: name)!
    try! archive.addComputePipelineFunctions(descriptor: d)
}
try! archive.serialize(to: URL(fileURLWithPath: args[2]))
print("ok")
