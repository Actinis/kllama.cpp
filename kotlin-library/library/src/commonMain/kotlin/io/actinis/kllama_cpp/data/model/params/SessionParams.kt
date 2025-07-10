package io.actinis.kllama_cpp.data.model.params

data class SessionParams(
    val modelPath: String,
    val mmprojPath: String = "",
    val contextSize: Int = 4096,
    val batch: Int = 4096,
    val gpuLayers: Int = 0,
    val mmprojUseGpu: Boolean = false,
    val threads: Int = 6,
    val verbosity: Int = 1,
    val sampling: SamplingParams = SamplingParams(),
)