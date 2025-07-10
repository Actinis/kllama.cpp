#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <cctype>

#include "KLlama.h"
#include "logging/logging.h"

#define LOG_TAG "KLlamaCPPDemo"

std::vector<uint8_t> readImageFile(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open image file: " + path);
    }

    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size <= 0) {
        throw std::runtime_error("Image file is empty or invalid: " + path);
    }

    std::vector<uint8_t> buffer(size);
    file.read(reinterpret_cast<char *>(buffer.data()), size);

    if (file.fail() || file.gcount() != size) {
        throw std::runtime_error("Failed to read image file completely: " + path);
    }

    return buffer;
}

std::string getImageFormat(const std::string &path) {
    if (const size_t dotPos = path.find_last_of('.'); dotPos != std::string::npos) {
        std::string ext = path.substr(dotPos + 1);
        std::ranges::transform(ext, ext.begin(), ::tolower);
        return ext;
    }
    return "png"; // default
}

void print_usage(int argc, char **argv) {
    (void) argc;
    printf("Usage: %s -m <model.gguf> -p <prompt> [options]\n\n", argv[0]);
    printf("Options:\n");
    printf("  -m, --model <path>         Path to the GGUF language model file (required).\n");
    printf("  -p, --prompt <text>        The text prompt to process (required).\n");
    printf("  --mmproj <path>            Path to the GGUF multimodal projector file (optional, for image support).\n");
    printf("  --image <path>             Path to an image file. Can be used multiple times. Requires --mmproj.\n");
    printf("  -t, --threads <n>          Number of threads to use (default: 6).\n");
    printf("  --temperature <f>          Temperature for sampling (default: 0.7).\n");
    printf("  --top-p <f>                Top-p for sampling (default: 0.9).\n");
    printf("  --top-k <n>                Top-k for sampling (default: 40).\n");
    printf("  --min-p <f>                Min-p for sampling (default: 0.05).\n");
    printf("  --repeat-penalty <f>       Repeat penalty (default: 1.1).\n");
    printf("  --repeat-last-n <n>        Last n tokens to apply repeat penalty (default: 64).\n");
    printf("  --max-tokens <n>           Maximum tokens to generate (default: unlimited).\n");
    printf("  --validate-model           Validate model file without full initialization.\n");
    printf("  -h, --help                 Show this help message.\n");
    printf("\nExample:\n");
    printf("  %s -m model.gguf -p \"Hello, how are you?\"\n", argv[0]);
    printf("  %s -m model.gguf -p \"What do you see?\" --image photo.jpg --mmproj vision.gguf\n", argv[0]);
    printf("  %s -m model.gguf -p \"Tell me a story\" --temperature 0.8 --top-p 0.95 --max-tokens 500\n", argv[0]);
    printf("  %s --validate-model -m model.gguf\n", argv[0]);
}

int main(int argc, char **argv) {
    if (argc <= 1) {
        print_usage(argc, argv);
        return EXIT_FAILURE;
    }

    SessionParams params;
    params.gpuLayers = -1;
    params.mmprojUseGpu = true;
    std::string prompt;
    std::vector<std::string> image_paths;
    bool validateOnly = false;

    // --- Argument Parsing ---
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return EXIT_SUCCESS;
        }
        if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) params.modelPath = argv[++i];
        } else if (arg == "--mmproj") {
            if (i + 1 < argc) params.mmprojPath = argv[++i];
        } else if (arg == "-p" || arg == "--prompt") {
            if (i + 1 < argc) prompt = argv[++i];
        } else if (arg == "--image") {
            if (i + 1 < argc) image_paths.emplace_back(argv[++i]);
        } else if (arg == "-t" || arg == "--threads") {
            if (i + 1 < argc) params.threads = std::stoi(argv[++i]);
        } else if (arg == "--temperature") {
            if (i + 1 < argc) params.sampling.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-p") {
            if (i + 1 < argc) params.sampling.topP = std::stof(argv[++i]);
        } else if (arg == "--top-k") {
            if (i + 1 < argc) params.sampling.topK = std::stoi(argv[++i]);
        } else if (arg == "--min-p") {
            if (i + 1 < argc) params.sampling.minP = std::stof(argv[++i]);
        } else if (arg == "--repeat-penalty") {
            if (i + 1 < argc) params.sampling.repeatPenalty = std::stof(argv[++i]);
        } else if (arg == "--repeat-last-n") {
            if (i + 1 < argc) params.sampling.repeatLastN = std::stoi(argv[++i]);
        } else if (arg == "--max-tokens") {
            if (i + 1 < argc) params.sampling.nPredict = std::stoi(argv[++i]);
        } else if (arg == "--validate-model") {
            validateOnly = true;
        } else {
            LOG_ERROR(LOG_TAG, "Unknown argument: %s", arg.c_str());
            print_usage(argc, argv);
            return EXIT_FAILURE;
        }
    }

    // --- Argument Validation ---
    if (params.modelPath.empty()) {
        LOG_ERROR(LOG_TAG, "Missing required argument: --model is required.");
        print_usage(argc, argv);
        return EXIT_FAILURE;
    }

    // --- Model Validation Mode ---
    if (validateOnly) {
        LOG_INFO(LOG_TAG, "Validating model: %s", params.modelPath.c_str());

        auto modelValidation = KLlama::validateModel(params.modelPath);
        if (modelValidation.isError()) {
            LOG_ERROR(LOG_TAG, "Model validation failed: %s", modelValidation.errorMessage.c_str());
            return EXIT_FAILURE;
        }

        const auto &info = modelValidation.value;
        LOG_INFO(LOG_TAG, "Model validation successful!");
        LOG_INFO(LOG_TAG, "  Name: %s", info.name.c_str());
        LOG_INFO(LOG_TAG, "  Parameters: %lld", info.parameterCount);
        LOG_INFO(LOG_TAG, "  Context size: %d", info.contextSize);
        LOG_INFO(LOG_TAG, "  Supports vision: %s", info.supportsVision ? "yes" : "no");

        if (!params.mmprojPath.empty()) {
            LOG_INFO(LOG_TAG, "Validating multimodal projector: %s", params.mmprojPath.c_str());
            if (auto mmprojValidation = KLlama::validateMmproj(params.mmprojPath); mmprojValidation.isError()) {
                LOG_ERROR(LOG_TAG, "Multimodal projector validation failed: %s", mmprojValidation.errorMessage.c_str());
                return EXIT_FAILURE;
            }
            LOG_INFO(LOG_TAG, "Multimodal projector validation successful!");
        }

        return EXIT_SUCCESS;
    }

    // --- Regular Generation Mode ---
    if (prompt.empty()) {
        LOG_ERROR(LOG_TAG, "Missing required argument: --prompt is required for generation.");
        print_usage(argc, argv);
        return EXIT_FAILURE;
    }

    if (!image_paths.empty() && params.mmprojPath.empty()) {
        LOG_ERROR(LOG_TAG, "Error: --image requires --mmproj to be specified.");
        print_usage(argc, argv);
        return EXIT_FAILURE;
    }

    // --- KLlama Initialization ---
    LOG_INFO(LOG_TAG, "Initializing KLlama with model: %s", params.modelPath.c_str());
    if (!params.mmprojPath.empty()) {
        LOG_INFO(LOG_TAG, "Using multimodal projector: %s", params.mmprojPath.c_str());
    } else {
        LOG_INFO(LOG_TAG, "Running in text-only mode.");
    }

    // Create KLlama instance
    KLlama kllama;

    // Define progress callback
    auto progressCallback = [](const float progress, const std::string &stage) {
        LOG_INFO(LOG_TAG, "Progress: %.1f%% - %s", progress * 100.0f, stage.c_str());
    };

    // Initialize with progress tracking
    if (auto initResult = kllama.initialize(params, progressCallback); initResult.isError()) {
        LOG_ERROR(LOG_TAG, "KLlama initialization failed: %s", initResult.errorMessage.c_str());
        return EXIT_FAILURE;
    }

    // Get and display model info
    auto modelInfoResult = kllama.getModelInfo();
    if (modelInfoResult.isSuccess()) {
        const auto &info = modelInfoResult.value;
        LOG_INFO(LOG_TAG, "Model loaded successfully:");
        LOG_INFO(LOG_TAG, "  Name: %s", info.name.c_str());
        LOG_INFO(LOG_TAG, "  Parameters: %lld", info.parameterCount);
        LOG_INFO(LOG_TAG, "  Context size: %d", info.contextSize);
        LOG_INFO(LOG_TAG, "  Supports vision: %s", info.supportsVision ? "yes" : "no");
    }

    // --- Build Conversation ---
    std::vector<MultimodalMessage> conversation;
    MultimodalMessage userMessage;
    userMessage.role = MessageRole::User; // UPDATED
    userMessage.content = prompt;

    // Load images if provided
    if (!image_paths.empty()) {
        LOG_INFO(LOG_TAG, "Loading %zu image(s)...", image_paths.size());

        for (const auto &imagePath: image_paths) {
            try {
                LOG_INFO(LOG_TAG, "Loading image: %s", imagePath.c_str());

                ImageData imageData;
                imageData.data = readImageFile(imagePath);

                // Validate image data
                auto imageValidation = KLlama::validateImageData(imageData);
                if (imageValidation.isError()) {
                    LOG_ERROR(LOG_TAG, "Image validation failed for %s: %s",
                              imagePath.c_str(), imageValidation.errorMessage.c_str());
                    return EXIT_FAILURE;
                }

                LOG_INFO(LOG_TAG, "Successfully loaded image: %s (size: %zu bytes)",
                         imagePath.c_str(), imageData.data.size());

                userMessage.images.push_back(std::move(imageData));
            } catch (const std::exception &e) {
                LOG_ERROR(LOG_TAG, "Failed to load image %s: %s", imagePath.c_str(), e.what());
                return EXIT_FAILURE;
            }
        }
    }

    conversation.push_back(std::move(userMessage));

    // --- Generate Response ---
    LOG_INFO(LOG_TAG, "Starting generation with temperature=%.2f, top_p=%.2f, top_k=%d",
             params.sampling.temperature, params.sampling.topP, params.sampling.topK);

    // Define a callback to stream tokens to the console
    auto tokenCallback = [](const std::string &token) {
        std::cout << token << std::flush;
    };

    // Define progress callback for generation
    auto genProgressCallback = [](float progress, const std::string &stage) {
        if (progress < 0.6f) {
            // Only show progress for preparation stages
            LOG_INFO(LOG_TAG, "Generation progress: %.1f%% - %s", progress * 100.0f, stage.c_str());
        }
    };

    // Print conversation context
    std::cout << "\n--- Conversation ---" << std::endl;
    std::cout << "User: " << prompt;
    if (!image_paths.empty()) {
        std::cout << " [with " << image_paths.size() << " image(s)]";
    }
    std::cout << std::endl;
    std::cout << "Assistant: " << std::flush;

    // Generate the response
    auto responseResult = kllama.generateResponse(conversation, tokenCallback, genProgressCallback);

    if (responseResult.isError()) {
        std::cout << std::endl;
        LOG_ERROR(LOG_TAG, "Generation failed: %s", responseResult.errorMessage.c_str());
        return EXIT_FAILURE;
    }

    const std::string &response = responseResult.value;
    std::cout << std::endl;

    LOG_INFO(LOG_TAG, "Generation completed successfully. Response length: %zu characters", response.length());

    // Display generation statistics
    auto statsResult = kllama.getGenerationStats();
    if (statsResult.isSuccess()) {
        const auto &stats = statsResult.value;
        LOG_INFO(LOG_TAG, "Generation statistics:");
        LOG_INFO(LOG_TAG, "  Tokens generated: %d", stats.tokensGenerated);
        LOG_INFO(LOG_TAG, "  Time elapsed: %.2f seconds", stats.timeElapsed);
        LOG_INFO(LOG_TAG, "  Tokens per second: %d", stats.tokensPerSecond);
    }

    // Display memory usage
    auto memoryResult = kllama.getMemoryInfo();
    if (memoryResult.isSuccess()) {
        const auto &memory = memoryResult.value;
        LOG_INFO(LOG_TAG, "Memory usage:");
        LOG_INFO(LOG_TAG, "  Model memory: %zu MB", memory.modelMemoryMB);
        LOG_INFO(LOG_TAG, "  Context memory: %zu MB", memory.contextMemoryMB);
        LOG_INFO(LOG_TAG, "  Total memory: %zu MB", memory.totalMemoryMB);
    }

    std::cout << "\n--- End of Response ---" << std::endl;

    return EXIT_SUCCESS;
}