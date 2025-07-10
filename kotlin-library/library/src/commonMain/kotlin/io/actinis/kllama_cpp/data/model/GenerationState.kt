package io.actinis.kllama_cpp.data.model

enum class GenerationState {
    Idle,
    Initializing,
    TokenizingPrompt,
    ProcessingImages,
    Generating,
    Finished,
    Cancelled,
    Error,
}