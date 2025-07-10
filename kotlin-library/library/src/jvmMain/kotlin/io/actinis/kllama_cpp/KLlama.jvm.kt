@file:Suppress("EXPECT_ACTUAL_CLASSIFIERS_ARE_IN_BETA_WARNING")

package io.actinis.kllama_cpp

import co.touchlab.kermit.Logger
import io.actinis.kllama_cpp.data.model.CancellationToken
import io.actinis.kllama_cpp.data.model.callback.ProgressCallback
import io.actinis.kllama_cpp.data.model.callback.TokenCallback
import io.actinis.kllama_cpp.data.model.info.GenerationStats
import io.actinis.kllama_cpp.data.model.info.MemoryInfo
import io.actinis.kllama_cpp.data.model.info.ModelInfo
import io.actinis.kllama_cpp.data.model.message.ImageData
import io.actinis.kllama_cpp.data.model.message.MultimodalMessage
import io.actinis.kllama_cpp.data.model.params.SamplingParams
import io.actinis.kllama_cpp.data.model.params.SessionParams
import io.actinis.kllama_cpp.data.model.result.KLlamaResult

actual class KLlama {

    private val logger = Logger.withTag(LOG_TAG)

    /**
     * A pointer to the underlying C++ KLlama object.
     * It's internal to prevent misuse.
     */
    @Volatile
    private var nativeHandle: Long = 0

    actual fun initialize(
        params: SessionParams,
        progressCallback: ProgressCallback?,
        cancellationToken: CancellationToken?,
    ): KLlamaResult<Unit> {
        if (nativeHandle != 0L) {
            freeMemory()
        }
        // The native method will create the C++ object and set nativeHandle
        return initializeNative(
            params,
            progressCallback,
            cancellationToken,
        )
    }

    actual fun generateResponse(
        conversation: List<MultimodalMessage>,
        sampling: SamplingParams,
        tokenCallback: TokenCallback?,
        progressCallback: ProgressCallback?,
        cancellationToken: CancellationToken?,
    ): KLlamaResult<String> {
        return generateResponseNative(
            conversation.toTypedArray(),
            sampling,
            tokenCallback,
            progressCallback,
            cancellationToken,
        )
    }

    actual fun isInitialized(): Boolean {
        return nativeHandle != 0L
    }

    actual fun getModelInfo(): KLlamaResult<ModelInfo> {
        return getModelInfoNative()
    }

    actual fun getMemoryInfo(): KLlamaResult<MemoryInfo> {
        return getMemoryInfoNative()
    }

    actual fun getGenerationStats(): KLlamaResult<GenerationStats> {
        return getGenerationStatsNative()
    }

    actual fun close() {
        if (nativeHandle != 0L) {
            freeMemory()
            nativeHandle = 0
        }
    }

    private external fun getModelInfoNative(): KLlamaResult<ModelInfo>
    private external fun getMemoryInfoNative(): KLlamaResult<MemoryInfo>
    private external fun getGenerationStatsNative(): KLlamaResult<GenerationStats>
    private external fun freeMemory()

    private external fun initializeNative(
        params: SessionParams,
        progressCallback: ProgressCallback?,
        cancellationToken: CancellationToken?,
    ): KLlamaResult<Unit>

    private external fun generateResponseNative(
        conversation: Array<MultimodalMessage>,
        sampling: SamplingParams,
        tokenCallback: TokenCallback?,
        progressCallback: ProgressCallback?,
        cancellationToken: CancellationToken?,
    ): KLlamaResult<String>

    actual companion object {
        private const val LOG_TAG = "KLlama"

        actual fun validateImageData(imageData: ImageData): KLlamaResult<ByteArray> {
            return validateImageDataNative(imageData)
        }

        actual fun validateModel(modelPath: String): KLlamaResult<ModelInfo> {
            return validateModelNative(modelPath)
        }

        actual fun validateMmproj(mmprojPath: String): KLlamaResult<Unit> {
            return validateMmprojNative(mmprojPath)
        }

        external fun validateImageDataNative(imageData: ImageData): KLlamaResult<ByteArray>
        external fun validateModelNative(modelPath: String): KLlamaResult<ModelInfo>
        external fun validateMmprojNative(mmprojPath: String): KLlamaResult<Unit>

        init {
            System.loadLibrary("kllama_cpp_jni")
        }
    }
}