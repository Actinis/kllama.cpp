package io.actinis.kllama_cpp.data.model.result

enum class KLlamaError {
    None,
    ModelNotFound,
    ModelLoadFailed,
    ModelInvalid,
    MmprojNotFound,
    MmprojLoadFailed,
    MmprojInvalid,
    ContextInitFailed,
    InsufficientMemory,
    TokenizationFailed,
    EvaluationFailed,
    SamplingFailed,
    ImageProcessingFailed,
    InvalidParameters,
    NotInitialized,
    AlreadyInitialized,
    OperationCancelled,
    UnknownError,
}