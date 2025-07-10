package io.actinis.kllama_cpp.data.model.message

data class MultimodalMessage(
    val role: MessageRole,
    val content: String,
    val images: List<ImageData> = emptyList(),
)


