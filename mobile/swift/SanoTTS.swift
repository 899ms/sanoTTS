// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Ampixa
//
// The sanoTTS inference runtime is MIT; see LICENSE.MIT for the exact file
// list and why the split is sound. The repository as a whole is GPL-3.0,
// because the grapheme-to-phoneme layer embeds espeak-ng. This file does not.
import Foundation
import AVFoundation

/// Swift wrapper over the C API in sanotts.h.
///
/// The C layer already owns the arena and the weights, so this adds only two
/// things Swift callers actually want: `deinit` instead of a manual close,
/// and an `AVAudioPCMBuffer` instead of a raw float pointer.
public final class SanoTTS {
    public enum Failure: Error, LocalizedError {
        case open(String), synthesize(String)
        public var errorDescription: String? {
            switch self { case .open(let m), .synthesize(let m): return m }
        }
    }

    private let handle: OpaquePointer

    /// Weights are usually bundle resources:
    ///   Bundle.main.url(forResource: "front_q8", withExtension: "bin")!
    public init(frontURL: URL, decoderURL: URL) throws {
        guard let h = frontURL.withUnsafeFileSystemRepresentation({ f in
            decoderURL.withUnsafeFileSystemRepresentation { d in
                sanotts_open(f, d)
            }
        }) else { throw Failure.open("could not allocate handle") }
        let message = String(cString: sanotts_last_error(h))
        guard message == "ok" else { sanotts_close(h); throw Failure.open(message) }
        handle = h
    }

    deinit { sanotts_close(handle) }

    public var sampleRate: Double { Double(sanotts_sample_rate(handle)) }

    /// Fixed by default, so the same ids give the same waveform every time.
    /// The decoder is noise-fed; a different seed is a different, equally
    /// valid rendering of the same utterance.
    public func setSeed(_ seed: UInt64) { sanotts_set_seed(handle, seed) }

    /// Phoneme ids, not text -- this library does no G2P. See mobile/README.md.
    public func synthesize(phonemeIDs ids: [Int32]) throws -> [Float] {
        var pcm: UnsafeMutablePointer<Float>?
        var count: Int32 = 0
        let rc = ids.withUnsafeBufferPointer {
            sanotts_speak(handle, $0.baseAddress, Int32(ids.count), &pcm, &count)
        }
        guard rc == 0, let samples = pcm else {
            throw Failure.synthesize(String(cString: sanotts_last_error(handle)))
        }
        defer { sanotts_free_pcm(samples) }
        return Array(UnsafeBufferPointer(start: samples, count: Int(count)))
    }

    /// Ready to hand to an AVAudioPlayerNode.
    public func buffer(phonemeIDs ids: [Int32]) throws -> AVAudioPCMBuffer {
        let samples = try synthesize(phonemeIDs: ids)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: sampleRate,
                                         channels: 1),
              let buffer = AVAudioPCMBuffer(pcmFormat: format,
                                            frameCapacity: AVAudioFrameCount(samples.count))
        else { throw Failure.synthesize("could not create an AVAudioPCMBuffer") }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer {
            buffer.floatChannelData![0].update(from: $0.baseAddress!, count: samples.count)
        }
        return buffer
    }
}
