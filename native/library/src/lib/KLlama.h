#ifndef KLLAMA_H
#define KLLAMA_H

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <atomic>
#include <chrono>

#include "llama.h"
#include "mtmd.h"

enum class KLlamaError {
    None = 0,
    ModelNotFound = 1,
    ModelLoadFailed = 2,
    ModelInvalid = 3,
    MmprojNotFound = 4,
    MmprojLoadFailed = 5,
    MmprojInvalid = 6,
    ContextInitFailed = 7,
    InsufficientMemory = 8,
    TokenizationFailed = 9,
    EvaluationFailed = 10,
    SamplingFailed = 11,
    ImageProcessingFailed = 12,
    InvalidParameters = 13,
    NotInitialized = 14,
    AlreadyInitialized = 15,
    OperationCancelled = 16,
    UnknownError = 99
};

template<typename T>
struct KLlamaResult {
    T value;
    KLlamaError error;
    std::string errorMessage;

    explicit KLlamaResult(T val) : value(std::move(val)), error(KLlamaError::None) {
    }

    explicit KLlamaResult(const KLlamaError error, std::string msg = "")
        : value{}, error(error), errorMessage(std::move(msg)) {
    }

    [[nodiscard]] bool isSuccess() const { return error == KLlamaError::None; }
    [[nodiscard]] bool isError() const { return error != KLlamaError::None; }
};

// Specialized for void operations
template<>
struct KLlamaResult<void> {
    KLlamaError error;
    std::string errorMessage;

    KLlamaResult() : error(KLlamaError::None) {
    }

    explicit KLlamaResult(const KLlamaError error, std::string msg = "")
        : error(error), errorMessage(std::move(msg)) {
    }

    [[nodiscard]] bool isSuccess() const { return error == KLlamaError::None; }
    [[nodiscard]] bool isError() const { return error != KLlamaError::None; }
};

// Progress callback types
using ProgressCallback = std::function<void(float progress, const std::string &stage)>;
using TokenCallback = std::function<void(const std::string &token)>;

// Cancellation token
struct CancellationToken {
    std::atomic<bool> cancelled{false};

    void cancel() { cancelled = true; }
    [[nodiscard]] bool isCancelled() const { return cancelled.load(); }
    void reset() { cancelled = false; }
};

// Model information
struct ModelInfo {
    std::string name;
    std::string architecture;
    int64_t parameterCount;
    int32_t contextSize;
    bool supportsVision;
    std::vector<std::string> capabilities;
};

// Memory usage information
struct MemoryInfo {
    size_t modelMemoryMB;
    size_t contextMemoryMB;
    size_t totalMemoryMB;
    size_t availableMemoryMB;
};

// Generation state
enum class GenerationState {
    Idle,
    Initializing,
    TokenizingPrompt,
    ProcessingImages,
    Generating,
    Finished,
    Cancelled,
    Error
};

// Generation statistics
struct GenerationStats {
    int32_t tokensGenerated;
    int32_t tokensPerSecond;
    float timeElapsed;
    GenerationState state;
};

struct ImageData {
    std::vector<uint8_t> data;
};

// UPDATED: Role is now a type-safe enum
enum class MessageRole {
    User,
    Assistant,
    System
};

struct MultimodalMessage {
    MessageRole role;
    std::string content;
    std::vector<ImageData> images;
};

struct SamplingParams {
    float temperature = 0.7f;
    float topP = 0.9f;
    int32_t topK = 40;
    float minP = 0.05f;
    float typicalP = 1.0f;
    float repeatPenalty = 1.1f;
    int32_t repeatLastN = 64;
    float frequencyPenalty = 0.0f;
    float presencePenalty = 0.0f;
    int32_t nPredict = -1; // -1 = unlimited

    // Validation
    [[nodiscard]] KLlamaResult<void> validate() const;
};

struct SessionParams {
    std::string modelPath;
    std::string mmprojPath;
    int contextSize = 16000;
    int batch = 4096;
    int gpuLayers = 0;
    bool mmprojUseGpu = false;
    int threads = 6;
    int verbosity = 1;
    SamplingParams sampling;

    [[nodiscard]] KLlamaResult<void> validate() const;
};

class KLlama {
public:
    KLlama() = default;

    ~KLlama();

    // Initialization with progress callback
    KLlamaResult<void> initialize(
        const SessionParams &sessionParams,
        const ProgressCallback &progressCallback = nullptr,
        const CancellationToken *cancellationToken = nullptr
    );

    // Model validation (lightweight check before full initialization)
    static KLlamaResult<ModelInfo> validateModel(const std::string &modelPath);

    static KLlamaResult<void> validateMmproj(const std::string &mmprojPath);

    // Utility methods (public)
    static KLlamaResult<void> checkFileExists(const std::string &path);

    static std::string errorToString(KLlamaError error);

    static KLlamaResult<std::vector<uint8_t> > validateImageData(const ImageData &imageData);

    // State queries
    bool isInitialized() const { return initialized; }
    GenerationState getGenerationState() const { return generationState; }

    KLlamaResult<ModelInfo> getModelInfo() const;

    KLlamaResult<MemoryInfo> getMemoryInfo() const;

    KLlamaResult<GenerationStats> getGenerationStats() const;

    // Generation with enhanced error handling and cancellation
    KLlamaResult<std::string> generateResponse(
        const std::vector<MultimodalMessage> &conversation,
        const TokenCallback &tokenCallback = nullptr,
        const ProgressCallback &progressCallback = nullptr,
        CancellationToken *cancellationToken = nullptr
    );

    KLlamaResult<std::string> generateResponse(
        const std::vector<MultimodalMessage> &conversation,
        const SamplingParams &samplingOverride,
        const TokenCallback &tokenCallback = nullptr,
        const ProgressCallback &progressCallback = nullptr,
        CancellationToken *cancellationToken = nullptr
    );

    // Memory management
    KLlamaResult<void> freeMemory();

    KLlamaResult<void> reset() const;

private:
    // Core state
    bool initialized = false;
    SessionParams params;
    GenerationState generationState = GenerationState::Idle;

    // llama.cpp objects
    mtmd::context_ptr visionContext;
    llama_model *model = nullptr;
    llama_context *llamaContext = nullptr;
    llama_sampler *sampler = nullptr;
    llama_batch batch{};

    // Generation statistics
    mutable GenerationStats currentStats{};
    std::chrono::high_resolution_clock::time_point generationStartTime;

    // Helper methods
    KLlamaResult<void> initializeModel(const ProgressCallback &progressCallback,
                                       const CancellationToken *cancellationToken);

    KLlamaResult<void> initializeVision(const ProgressCallback &progressCallback,
                                        const CancellationToken *cancellationToken);

    KLlamaResult<void> configureSampler(const SamplingParams &samplingParams);

    KLlamaResult<std::string> generateResponseInternal(
        const std::vector<MultimodalMessage> &conversation,
        const SamplingParams &samplingParams,
        const TokenCallback &tokenCallback,
        const ProgressCallback &progressCallback,
        CancellationToken *cancellationToken
    );

    // REMOVED: This is no longer needed.
    // static std::string buildConversationPrompt(const std::vector<MultimodalMessage> &conversation);

    static std::vector<ImageData> extractAllImages(const std::vector<MultimodalMessage> &conversation);

    void updateGenerationStats() const;

    void setGenerationState(GenerationState state);

    // Error handling helpers
    KLlamaResult<void> checkInitialized() const;
};

#endif