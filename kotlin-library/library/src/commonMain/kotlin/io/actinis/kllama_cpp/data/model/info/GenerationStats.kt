package io.actinis.kllama_cpp.data.model.info

import io.actinis.kllama_cpp.data.model.GenerationState
import io.actinis.kllama_cpp.data.model.params.SamplingParams

data class GenerationStats(
    val tokensGenerated: Int,
    val tokensPerSecond: Int,
    val timeElapsed: Float,
    val state: GenerationState,
    val defaultSampling: SamplingParams,
)