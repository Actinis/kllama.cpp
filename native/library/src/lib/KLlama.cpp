#include "KLlama.h"

#include <iostream>
#include <utility>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

static std::string tokenToString(const llama_model *model, const llama_token token) {
    std::vector<char> buf(32, 0);
    const auto *vocab = llama_model_get_vocab(model);
    const auto n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    if (n < 0) {
        buf.resize(-n);
        llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, true);
    }
    return {buf.data(), static_cast<size_t>(n)};
}

// Error handling implementations
std::string KLlama::errorToString(KLlamaError error) {
    switch (error) {
        case KLlamaError::None: return "Success";
        case KLlamaError::ModelNotFound: return "Model file not found";
        case KLlamaError::ModelLoadFailed: return "Failed to load model";
        case KLlamaError::ModelInvalid: return "Invalid model format";
        case KLlamaError::MmprojNotFound: return "Multimodal projector file not found";
        case KLlamaError::MmprojLoadFailed: return "Failed to load multimodal projector";
        case KLlamaError::MmprojInvalid: return "Invalid multimodal projector format";
        case KLlamaError::ContextInitFailed: return "Failed to initialize context";
        case KLlamaError::InsufficientMemory: return "Insufficient memory";
        case KLlamaError::TokenizationFailed: return "Text tokenization failed";
        case KLlamaError::EvaluationFailed: return "Model evaluation failed";
        case KLlamaError::SamplingFailed: return "Token sampling failed";
        case KLlamaError::ImageProcessingFailed: return "Image processing failed";
        case KLlamaError::InvalidParameters: return "Invalid parameters";
        case KLlamaError::NotInitialized: return "KLlama not initialized";
        case KLlamaError::AlreadyInitialized: return "KLlama already initialized";
        case KLlamaError::OperationCancelled: return "Operation was cancelled";
        case KLlamaError::UnknownError: return "Unknown error";
    }
    return "Unknown error code";
}

KLlamaResult<void> KLlama::checkFileExists(const std::string &path) {
    if (!std::filesystem::exists(path)) {
        return KLlamaResult<void>(KLlamaError::ModelNotFound, "File not found: " + path);
    }
    return {};
}

KLlamaResult<void> KLlama::checkInitialized() const {
    if (!initialized) {
        return KLlamaResult<void>(KLlamaError::NotInitialized, "KLlama must be initialized before use");
    }
    return {};
}

// Parameter validation
KLlamaResult<void> SamplingParams::validate() const {
    if (temperature < 0.0f || temperature > 2.0f) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "Temperature must be between 0.0 and 2.0");
    }
    if (topP < 0.0f || topP > 1.0f) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "top_p must be between 0.0 and 1.0");
    }
    if (topK < 0) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "top_k must be non-negative");
    }
    if (minP < 0.0f || minP > 1.0f) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "min_p must be between 0.0 and 1.0");
    }
    if (repeatPenalty < 0.0f) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "repeat_penalty must be non-negative");
    }
    if (repeatLastN < 0) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "repeat_last_n must be non-negative");
    }
    return {};
}

KLlamaResult<void> SessionParams::validate() const {
    if (modelPath.empty()) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "Model path cannot be empty");
    }

    auto fileCheck = KLlama::checkFileExists(modelPath);
    if (fileCheck.isError()) {
        return fileCheck;
    }

    if (!mmprojPath.empty()) {
        auto mmprojCheck = KLlama::checkFileExists(mmprojPath);
        if (mmprojCheck.isError()) {
            return KLlamaResult<void>(KLlamaError::MmprojNotFound,
                                      "Multimodal projector file not found: " + mmprojPath);
        }
    }

    if (contextSize <= 0) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "Context size must be positive");
    }
    if (batch <= 0) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "Batch size must be positive");
    }
    if (threads <= 0) {
        return KLlamaResult<void>(KLlamaError::InvalidParameters, "Thread count must be positive");
    }

    return sampling.validate();
}

// Constructor and destructor
KLlama::~KLlama() {
    if (initialized) {
        freeMemory();
    }
}

KLlamaResult<void> KLlama::freeMemory() {
    if (batch.token) {
        llama_batch_free(batch);
        batch = {};
    }
    if (sampler) {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }
    if (llamaContext) {
        llama_free(llamaContext);
        llamaContext = nullptr;
    }
    if (model) {
        llama_model_free(model);
        model = nullptr;
    }
    if (initialized) {
        llama_backend_free();
        initialized = false;
    }
    visionContext.reset();

    setGenerationState(GenerationState::Idle);

    return {};
}

// Initialization with progress and cancellation
KLlamaResult<void> KLlama::initialize(
    const SessionParams &sessionParams,
    const ProgressCallback &progressCallback,
    const CancellationToken *cancellationToken
) {
    if (initialized) {
        return KLlamaResult<void>(KLlamaError::AlreadyInitialized);
    }

    // Validate parameters
    auto validation = sessionParams.validate();
    if (validation.isError()) {
        return validation;
    }

    params = sessionParams;
    setGenerationState(GenerationState::Initializing);

    if (progressCallback) {
        progressCallback(0.0f, "Initializing backend");
    }

    // Check cancellation
    if (cancellationToken && cancellationToken->isCancelled()) {
        setGenerationState(GenerationState::Cancelled);
        return KLlamaResult<void>(KLlamaError::OperationCancelled);
    }

    llama_backend_init();
    batch = llama_batch_init(1, 0, 1);

    // Initialize model
    if (auto modelResult = initializeModel(progressCallback, cancellationToken); modelResult.isError()) {
        freeMemory();
        return modelResult;
    }

    // Initialize vision if needed
    if (!params.mmprojPath.empty()) {
        if (auto visionResult = initializeVision(progressCallback, cancellationToken); visionResult.isError()) {
            freeMemory();
            return visionResult;
        }
    }

    initialized = true;
    setGenerationState(GenerationState::Idle);

    if (progressCallback) {
        progressCallback(1.0f, "Initialization complete");
    }

    return {};
}

KLlamaResult<void> KLlama::initializeModel(
    const ProgressCallback &progressCallback,
    const CancellationToken *cancellationToken
) {
    if (progressCallback) {
        progressCallback(0.1f, "Loading model");
    }

    auto modelParams = llama_model_default_params();
    model = llama_model_load_from_file(params.modelPath.c_str(), modelParams);
    if (!model) {
        return KLlamaResult<void>(KLlamaError::ModelLoadFailed, "Failed to load model from: " + params.modelPath);
    }

    if (cancellationToken && cancellationToken->isCancelled()) {
        return KLlamaResult<void>(KLlamaError::OperationCancelled);
    }

    if (progressCallback) {
        progressCallback(0.4f, "Initializing context");
    }

    llama_context_params contextParams = llama_context_default_params();
    contextParams.n_ctx = params.contextSize;
    contextParams.n_batch = params.batch;
    contextParams.n_threads = params.threads;
    contextParams.n_threads_batch = params.threads;

    llamaContext = llama_init_from_model(model, contextParams);
    if (!llamaContext) {
        return KLlamaResult<void>(KLlamaError::ContextInitFailed, "Failed to initialize llama context");
    }

    if (progressCallback) {
        progressCallback(0.6f, "Model loaded successfully");
    }

    return {};
}

KLlamaResult<void> KLlama::initializeVision(
    const ProgressCallback &progressCallback,
    const CancellationToken *cancellationToken
) {
    if (progressCallback) {
        progressCallback(0.7f, "Loading vision model");
    }

    auto multimodalContextParams = mtmd_context_params_default();
    multimodalContextParams.use_gpu = params.mmprojUseGpu;
    multimodalContextParams.n_threads = params.threads;
    multimodalContextParams.verbosity = params.verbosity > 1 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;

    visionContext.reset(mtmd_init_from_file(params.mmprojPath.c_str(), model, multimodalContextParams));
    if (!visionContext) {
        return KLlamaResult<void>(KLlamaError::MmprojLoadFailed,
                                  "Failed to load vision model from: " + params.mmprojPath);
    }

    if (cancellationToken && cancellationToken->isCancelled()) {
        return KLlamaResult<void>(KLlamaError::OperationCancelled);
    }

    if (progressCallback) {
        progressCallback(0.9f, "Vision model loaded successfully");
    }

    return {};
}

// Model validation
KLlamaResult<ModelInfo> KLlama::validateModel(const std::string &modelPath) {
    if (const auto fileCheck = checkFileExists(modelPath); fileCheck.isError()) {
        return KLlamaResult<ModelInfo>(fileCheck.error, fileCheck.errorMessage);
    }

    // Quick validation by attempting to load model metadata
    llama_backend_init();
    const auto modelParams = llama_model_default_params();

    llama_model *tempModel = llama_model_load_from_file(modelPath.c_str(), modelParams);
    if (!tempModel) {
        llama_backend_free();
        return KLlamaResult<ModelInfo>(KLlamaError::ModelInvalid, "Invalid model format");
    }

    ModelInfo info;

    // Get model description with proper buffer
    char desc_buffer[256];
    if (const int32_t descLen = llama_model_desc(tempModel, desc_buffer, sizeof(desc_buffer)); descLen > 0) {
        info.name = std::string(desc_buffer, descLen);
    } else {
        info.name = "Unknown Model";
    }

    info.parameterCount = static_cast<int64_t>(llama_model_n_params(tempModel));
    info.contextSize = llama_model_n_ctx_train(tempModel);
    info.supportsVision = false; // Will be true if mmproj is loaded

    llama_model_free(tempModel);
    llama_backend_free();

    return KLlamaResult(std::move(info));
}

KLlamaResult<void> KLlama::validateMmproj(const std::string &mmprojPath) {
    auto fileCheck = checkFileExists(mmprojPath);
    if (fileCheck.isError()) {
        return KLlamaResult<void>(KLlamaError::MmprojNotFound, fileCheck.errorMessage);
    }

    // Quick validation would require loading both model and mmproj
    // For now, just check file existence and basic format
    std::ifstream file(mmprojPath, std::ios::binary);
    if (!file.is_open()) {
        return KLlamaResult<void>(KLlamaError::MmprojInvalid, "Cannot open mmproj file");
    }

    // Basic GGUF header check
    char header[4];
    file.read(header, 4);
    if (std::string(header, 4) != "GGUF") {
        return KLlamaResult<void>(KLlamaError::MmprojInvalid, "Invalid mmproj format - not a GGUF file");
    }

    return {};
}

// State queries
KLlamaResult<ModelInfo> KLlama::getModelInfo() const {
    if (const auto initCheck = checkInitialized(); initCheck.isError()) {
        return KLlamaResult<ModelInfo>(initCheck.error, initCheck.errorMessage);
    }

    ModelInfo info;

    // Get model description with proper buffer
    char desc_buffer[256];
    if (const int32_t descLen = llama_model_desc(model, desc_buffer, sizeof(desc_buffer)); descLen > 0) {
        info.name = std::string(desc_buffer, descLen);
    } else {
        info.name = "Unknown Model";
    }

    info.parameterCount = static_cast<int64_t>(llama_model_n_params(model));
    info.contextSize = llama_model_n_ctx_train(model);
    info.supportsVision = (visionContext != nullptr);

    // Add capabilities
    info.capabilities.emplace_back("text_generation");
    if (info.supportsVision) {
        info.capabilities.emplace_back("vision");
        info.capabilities.emplace_back("multimodal");
    }

    return KLlamaResult(std::move(info));
}

KLlamaResult<MemoryInfo> KLlama::getMemoryInfo() const {
    auto initCheck = checkInitialized();
    if (initCheck.isError()) {
        return KLlamaResult<MemoryInfo>(initCheck.error, initCheck.errorMessage);
    }

    MemoryInfo info{};
    info.modelMemoryMB = llama_model_size(model) / (1024 * 1024);

    info.contextMemoryMB = llama_state_get_size(llamaContext) / (1024 * 1024);
    info.totalMemoryMB = info.modelMemoryMB + info.contextMemoryMB;


    return KLlamaResult(info);
}

KLlamaResult<GenerationStats> KLlama::getGenerationStats() const {
    if (const auto initCheck = checkInitialized(); initCheck.isError()) {
        return KLlamaResult<GenerationStats>(initCheck.error, initCheck.errorMessage);
    }

    return KLlamaResult(currentStats);
}

// Image validation
KLlamaResult<std::vector<uint8_t> > KLlama::validateImageData(const ImageData &imageData) {
    if (imageData.data.empty()) {
        return KLlamaResult<std::vector<uint8_t> >(KLlamaError::ImageProcessingFailed, "Image data is empty");
    }

    // Basic format validation
    if (imageData.data.size() < 8) {
        return KLlamaResult<std::vector<uint8_t> >(KLlamaError::ImageProcessingFailed, "Image data too small");
    }

    // Check for common image headers
    const auto &data = imageData.data;
    bool validFormat = false;

    // PNG header
    if (data.size() >= 8 && data[0] == 0x89 && data[1] == 0x50 && data[2] == 0x4E && data[3] == 0x47) {
        validFormat = true;
    }
    // JPEG header
    else if (data.size() >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
        validFormat = true;
    }
    // BMP header
    else if (data.size() >= 2 && data[0] == 0x42 && data[1] == 0x4D) {
        validFormat = true;
    }

    if (!validFormat) {
        return KLlamaResult<std::vector<uint8_t> >(KLlamaError::ImageProcessingFailed, "Unsupported image format");
    }

    return KLlamaResult(imageData.data);
}

// Sampler configuration
KLlamaResult<void> KLlama::configureSampler(const SamplingParams &samplingParams) {
    auto validation = samplingParams.validate();
    if (validation.isError()) {
        return validation;
    }

    // Free existing sampler
    if (sampler) {
        llama_sampler_free(sampler);
        sampler = nullptr;
    }

    // Create new sampler chain
    sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (!sampler) {
        return KLlamaResult<void>(KLlamaError::SamplingFailed, "Failed to create sampler chain");
    }

    // Add penalty samplers first (if repeat penalty is enabled)
    if (samplingParams.repeatPenalty != 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
                                    samplingParams.repeatLastN,
                                    samplingParams.repeatPenalty,
                                    samplingParams.frequencyPenalty,
                                    samplingParams.presencePenalty
                                ));
    }

    // If temperature is very low, use greedy sampling only
    if (samplingParams.temperature <= 0.01f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        return {};
    }

    // Add top-k sampling (if enabled)
    if (samplingParams.topK > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(samplingParams.topK));
    }

    // Add typical sampling (if enabled)
    if (samplingParams.typicalP < 1.0f && samplingParams.typicalP > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_typical(samplingParams.typicalP, 1));
    }

    // Add top-p sampling (if enabled)
    if (samplingParams.topP < 1.0f && samplingParams.topP > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(samplingParams.topP, 1));
    }

    // Add min-p sampling (if enabled)
    if (samplingParams.minP > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_min_p(samplingParams.minP, 1));
    }

    // Add temperature sampling
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(samplingParams.temperature));

    // Add the final multinomial sampler
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    return {};
}

// Generation methods
KLlamaResult<std::string> KLlama::generateResponse(
    const std::vector<MultimodalMessage> &conversation,
    const TokenCallback &tokenCallback,
    const ProgressCallback &progressCallback,
    CancellationToken *cancellationToken
) {
    return generateResponseInternal(conversation, params.sampling, tokenCallback, progressCallback, cancellationToken);
}

KLlamaResult<std::string> KLlama::generateResponse(
    const std::vector<MultimodalMessage> &conversation,
    const SamplingParams &samplingOverride,
    const TokenCallback &tokenCallback,
    const ProgressCallback &progressCallback,
    CancellationToken *cancellationToken
) {
    return generateResponseInternal(conversation, samplingOverride, tokenCallback, progressCallback, cancellationToken);
}

KLlamaResult<std::string> KLlama::generateResponseInternal(
    const std::vector<MultimodalMessage> &conversation,
    const SamplingParams &samplingParams,
    const TokenCallback &tokenCallback,
    const ProgressCallback &progressCallback,
    CancellationToken *cancellationToken
) {
    if (auto initCheck = checkInitialized(); initCheck.isError()) {
        return KLlamaResult<std::string>(initCheck.error, initCheck.errorMessage);
    }

    if (generationState != GenerationState::Idle) {
        return KLlamaResult<std::string>(KLlamaError::InvalidParameters, "Generation already in progress");
    }

    // Validate conversation
    if (conversation.empty()) {
        return KLlamaResult<std::string>(KLlamaError::InvalidParameters, "Conversation cannot be empty");
    }

    // Validate images if present
    std::vector<ImageData> allImages = extractAllImages(conversation);
    for (const auto &image: allImages) {
        if (auto imageValidation = validateImageData(image); imageValidation.isError()) {
            return KLlamaResult<std::string>(imageValidation.error, imageValidation.errorMessage);
        }
    }

    // Check if vision is needed but not available
    if (!allImages.empty() && !visionContext) {
        return KLlamaResult<std::string>(KLlamaError::InvalidParameters,
                                         "Images provided but multimodal projector not loaded");
    }

    // Configure sampler
    auto samplerResult = configureSampler(samplingParams);
    if (samplerResult.isError()) {
        return KLlamaResult<std::string>(samplerResult.error, samplerResult.errorMessage);
    }

    // Initialize generation state
    setGenerationState(GenerationState::Initializing);
    currentStats = {};
    currentStats.sampling = samplingParams;
    generationStartTime = std::chrono::high_resolution_clock::now();

    // Reset context
    if (auto resetResult = reset(); resetResult.isError()) {
        setGenerationState(GenerationState::Error);
        return KLlamaResult<std::string>(resetResult.error, resetResult.errorMessage);
    }

    // 1. Convert our MultimodalMessage to llama_chat_message
    std::vector<llama_chat_message> chatMessages;
    chatMessages.reserve(conversation.size());
    for (const auto &msg: conversation) {
        const char *role_str;
        switch (msg.role) {
            case MessageRole::User:
                role_str = "user";
                break;
            case MessageRole::Assistant:
                role_str = "assistant";
                break;
            case MessageRole::System:
                role_str = "system";
                break;
            default: // Should not happen
                role_str = "user";
                break;
        }
        chatMessages.push_back({role_str, msg.content.c_str()});
    }

    // 2. Apply the chat template from the model
    std::vector<char> promptBuffer(params.contextSize);

    const int32_t prompt_len = llama_chat_apply_template(
        nullptr, // Use template from model
        chatMessages.data(),
        chatMessages.size(),
        true, // Add generation prompt for assistant
        promptBuffer.data(),
        static_cast<int32_t>(promptBuffer.size())
    );

    if (prompt_len < 0) {
        return KLlamaResult<std::string>(KLlamaError::TokenizationFailed,
                                         "Failed to apply chat template. Prompt may be too long or template invalid.");
    }

    std::string fullPrompt(promptBuffer.data(), prompt_len);

    try {
        llama_pos past = 0;
        // Check cancellation
        if (cancellationToken && cancellationToken->isCancelled()) {
            setGenerationState(GenerationState::Cancelled);
            return KLlamaResult<std::string>(KLlamaError::OperationCancelled);
        }

        if (!allImages.empty()) {
            // Multimodal processing
            setGenerationState(GenerationState::ProcessingImages);
            if (progressCallback) {
                progressCallback(0.1f, "Processing images");
            }

            // Prepend image markers to prompt
            std::string imageMarkers;
            for (size_t i = 0; i < allImages.size(); ++i) {
                imageMarkers += mtmd_default_marker();
            }
            fullPrompt = imageMarkers + "\n" + fullPrompt;

            // Create bitmaps
            mtmd::bitmaps bitmaps;
            for (const auto &imageData: allImages) {
                mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(
                    visionContext.get(),
                    imageData.data.data(),
                    imageData.data.size()
                ));
                if (!bmp.ptr) {
                    setGenerationState(GenerationState::Error);
                    return KLlamaResult<std::string>(KLlamaError::ImageProcessingFailed,
                                                     "Failed to create bitmap from image data");
                }
                bitmaps.entries.push_back(std::move(bmp));
            }

            setGenerationState(GenerationState::TokenizingPrompt);
            if (progressCallback) {
                progressCallback(0.3f, "Tokenizing multimodal prompt");
            }

            mtmd_input_text textInput;
            textInput.text = fullPrompt.c_str();
            textInput.add_special = true;
            textInput.parse_special = true;

            const mtmd::input_chunks chunks(mtmd_input_chunks_init());
            auto bitmapsCPointer = bitmaps.c_ptr();
            const auto tokenizationResult = mtmd_tokenize(
                visionContext.get(),
                chunks.ptr.get(),
                &textInput,
                bitmapsCPointer.data(),
                bitmapsCPointer.size()
            );

            if (tokenizationResult != 0) {
                setGenerationState(GenerationState::Error);
                return KLlamaResult<std::string>(KLlamaError::TokenizationFailed,
                                                 "Failed to tokenize multimodal input");
            }

            // Check cancellation
            if (cancellationToken && cancellationToken->isCancelled()) {
                setGenerationState(GenerationState::Cancelled);
                return KLlamaResult<std::string>(KLlamaError::OperationCancelled);
            }

            if (progressCallback) {
                progressCallback(0.5f, "Evaluating multimodal prompt");
            }

            llama_pos newPast = 0;
            if (mtmd_helper_eval_chunks(
                visionContext.get(),
                llamaContext,
                chunks.ptr.get(),
                past,
                0,
                params.batch,
                true,
                &newPast
            )) {
                setGenerationState(GenerationState::Error);
                return KLlamaResult<std::string>(KLlamaError::EvaluationFailed,
                                                 "Failed to evaluate multimodal prompt");
            }
            past = newPast;
        } else {
            // Text-only processing
            setGenerationState(GenerationState::TokenizingPrompt);
            if (progressCallback) {
                progressCallback(0.2f, "Tokenizing text prompt");
            }

            std::vector<llama_token> prompt_tokens;
            prompt_tokens.resize(fullPrompt.length() + 2);

            const auto *vocab = llama_model_get_vocab(model);
            const auto tokensNumber = llama_tokenize(
                vocab,
                fullPrompt.c_str(),
                static_cast<int32_t>(fullPrompt.length()),
                prompt_tokens.data(),
                static_cast<int32_t>(prompt_tokens.size()),
                false, // Do not add BOS token, template should handle it
                true // Parse special tokens from the template
            );

            if (tokensNumber < 0) {
                setGenerationState(GenerationState::Error);
                return KLlamaResult<std::string>(KLlamaError::TokenizationFailed,
                                                 "Failed to tokenize text prompt: prompt too long");
            }

            prompt_tokens.resize(tokensNumber);

            if (progressCallback) {
                progressCallback(0.4f, "Evaluating text prompt");
            }

            auto textBatch = llama_batch_init(tokensNumber, 0, 1);
            textBatch.n_tokens = tokensNumber;

            for (int i = 0; i < tokensNumber; ++i) {
                textBatch.token[i] = prompt_tokens[i];
                textBatch.pos[i] = past + i;
                textBatch.n_seq_id[i] = 1;
                textBatch.seq_id[i][0] = 0;
                textBatch.logits[i] = false;
            }
            textBatch.logits[textBatch.n_tokens - 1] = true;

            if (llama_decode(llamaContext, textBatch)) {
                llama_batch_free(textBatch);
                setGenerationState(GenerationState::Error);
                return KLlamaResult<std::string>(KLlamaError::EvaluationFailed,
                                                 "Failed to evaluate text prompt");
            }

            past += tokensNumber;
            llama_batch_free(textBatch);
        }

        // Check cancellation before generation
        if (cancellationToken && cancellationToken->isCancelled()) {
            setGenerationState(GenerationState::Cancelled);
            return KLlamaResult<std::string>(KLlamaError::OperationCancelled);
        }

        // Start generation
        setGenerationState(GenerationState::Generating);
        if (progressCallback) {
            progressCallback(0.6f, "Generating response");
        }

        std::string response_text;
        int32_t tokenCount = 0;
        int32_t maxTokens = samplingParams.nPredict > 0 ? samplingParams.nPredict : 4096;

        while (generationState == GenerationState::Generating && tokenCount < maxTokens) {
            // Check cancellation
            if (cancellationToken && cancellationToken->isCancelled()) {
                setGenerationState(GenerationState::Cancelled);
                return KLlamaResult<std::string>(KLlamaError::OperationCancelled);
            }

            const auto id = llama_sampler_sample(sampler, llamaContext, -1);

            if (id == LLAMA_TOKEN_NULL) {
                setGenerationState(GenerationState::Error);
                return KLlamaResult<std::string>(KLlamaError::SamplingFailed,
                                                 "Sampler returned null token");
            }

            llama_sampler_accept(sampler, id);

            if (llama_vocab_is_eog(llama_model_get_vocab(model), id)) {
                break;
            }

            std::string token_str = tokenToString(model, id);
            response_text += token_str;

            if (tokenCallback) {
                tokenCallback(token_str);
            }

            // Update statistics
            currentStats.tokensGenerated = tokenCount + 1;
            updateGenerationStats();

            // Prepare batch for the next token
            batch.n_tokens = 1;
            batch.token[0] = id;
            batch.pos[0] = past;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = true;

            past++;
            tokenCount++;

            if (llama_decode(llamaContext, batch)) {
                setGenerationState(GenerationState::Error);
                return KLlamaResult<std::string>(KLlamaError::EvaluationFailed,
                                                 "Failed to decode token");
            }

            // Progress update
            if (progressCallback && samplingParams.nPredict > 0) {
                float progress = 0.6f + 0.4f * static_cast<float>(tokenCount) / static_cast<float>(maxTokens);
                progressCallback(progress, "Generating tokens");
            }
        }

        setGenerationState(GenerationState::Finished);
        if (progressCallback) {
            progressCallback(1.0f, "Generation complete");
        }

        return KLlamaResult(std::move(response_text));
    } catch (const std::exception &e) {
        setGenerationState(GenerationState::Error);
        return KLlamaResult<std::string>(KLlamaError::UnknownError, e.what());
    }
}

// Helper methods
std::vector<ImageData> KLlama::extractAllImages(const std::vector<MultimodalMessage> &conversation) {
    std::vector<ImageData> allImages;

    for (const auto &message: conversation) {
        for (const auto &image: message.images) {
            allImages.push_back(image);
        }
    }

    return allImages;
}

KLlamaResult<void> KLlama::reset() const {
    if (auto initCheck = checkInitialized(); initCheck.isError()) {
        return initCheck;
    }

    llama_memory_seq_rm(llama_get_memory(llamaContext), 0, -1, -1);
    return {};
}

void KLlama::setGenerationState(GenerationState state) {
    generationState = state;
    currentStats.state = state;
}

void KLlama::updateGenerationStats() const {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - generationStartTime);

    currentStats.timeElapsed = static_cast<float>(elapsed.count()) / 1000.0f;
    if (currentStats.timeElapsed > 0.0f) {
        currentStats.tokensPerSecond = static_cast<int32_t>(
            static_cast<float>(currentStats.tokensGenerated) / currentStats.timeElapsed);
    }
}
