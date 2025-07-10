package io.actinis.kllama_cpp.data.model.params

data class SamplingParams(
    val temperature: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Int = 40,
    val minP: Float = 0.05f,
    val typicalP: Float = 1.0f,
    val repeatPenalty: Float = 1.1f,
    val repeatLastN: Int = 64,
    val frequencyPenalty: Float = 0.0f,
    val presencePenalty: Float = 0.0f,
    val nPredict: Int = -1,
)