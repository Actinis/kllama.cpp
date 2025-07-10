package io.actinis.kllama_cpp.data.model.result

sealed interface KLlamaResult<T> {
    data class Success<T>(val value: T) : KLlamaResult<T>
    data class Error<T>(val code: KLlamaError, val message: String) : KLlamaResult<T>
}