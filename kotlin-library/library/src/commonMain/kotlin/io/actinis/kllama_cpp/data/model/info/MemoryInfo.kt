package io.actinis.kllama_cpp.data.model.info

data class MemoryInfo(
    val modelMemoryMB: Long,
    val contextMemoryMB: Long,
    val totalMemoryMB: Long,
    val availableMemoryMB: Long,
)