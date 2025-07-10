@file:Suppress("EXPECT_ACTUAL_CLASSIFIERS_ARE_IN_BETA_WARNING")

package io.actinis.kllama_cpp

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

/**
 * The main wrapper class for the native KLlamaCpp library.
 */
expect class KLlama {

    /**
     * Initializes the llama.cpp backend and loads the specified models.
     * This must be called before any other methods.
     *
     * @param params The session parameters.
     * @param progressCallback An optional callback to report loading progress.
     * @param cancellationToken An optional token to cancel initialization.
     * @return A [KLlamaResult] indicating success or failure.
     */
    fun initialize(
        params: SessionParams,
        progressCallback: ProgressCallback? = null,
        cancellationToken: CancellationToken? = null,
    ): KLlamaResult<Unit>


    /**
     * Generates a response based on a conversation history.
     *
     * @param conversation The list of messages forming the conversation context.
     * @param sampling The sampling parameters to use for this generation.
     * @param tokenCallback A callback to stream generated tokens as they become available.
     * @param progressCallback A callback to report generation progress (e.g., prompt processing).
     * @param cancellationToken A token to cancel the generation.
     * @return A [KLlamaResult] containing the full generated response on success.
     */
    fun generateResponse(
        conversation: List<MultimodalMessage>,
        sampling: SamplingParams = SamplingParams(),
        tokenCallback: TokenCallback? = null,
        progressCallback: ProgressCallback? = null,
        cancellationToken: CancellationToken? = null,
    ): KLlamaResult<String>

    fun isInitialized(): Boolean
    fun getModelInfo(): KLlamaResult<ModelInfo>
    fun getMemoryInfo(): KLlamaResult<MemoryInfo>
    fun getGenerationStats(): KLlamaResult<GenerationStats>

    fun close()

    companion object {
        fun validateModel(modelPath: String): KLlamaResult<ModelInfo>
        fun validateMmproj(mmprojPath: String): KLlamaResult<Unit>
        fun validateImageData(imageData: ImageData): KLlamaResult<ByteArray>
    }
}