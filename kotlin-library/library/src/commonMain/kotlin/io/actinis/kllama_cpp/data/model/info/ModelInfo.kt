package io.actinis.kllama_cpp.data.model.info

data class ModelInfo(
    val name: String,
    val architecture: String,
    val parameterCount: Long,
    val contextSize: Int,
    val supportsVision: Boolean,
    val capabilities: List<String>,
)