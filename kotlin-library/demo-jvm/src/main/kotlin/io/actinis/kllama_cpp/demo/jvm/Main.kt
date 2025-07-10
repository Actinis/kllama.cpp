package io.actinis.kllama_cpp.demo.jvm

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.main
import com.github.ajalt.clikt.parameters.options.*
import com.github.ajalt.clikt.parameters.types.float
import com.github.ajalt.clikt.parameters.types.int
import io.actinis.kllama_cpp.KLlama
import io.actinis.kllama_cpp.data.model.message.ImageData
import io.actinis.kllama_cpp.data.model.message.MessageRole
import io.actinis.kllama_cpp.data.model.message.MultimodalMessage
import io.actinis.kllama_cpp.data.model.params.SamplingParams
import io.actinis.kllama_cpp.data.model.params.SessionParams
import io.actinis.kllama_cpp.data.model.result.KLlamaResult
import java.io.File

class DemoCommand : CliktCommand(
    name = "kllama-cpp-jvm-demo",
) {

    private val model by option("-m", "--model", help = "Path to the GGUF language model file").required()
    private val prompt by option("-p", "--prompt", help = "The text prompt to process")
    private val mmproj by option("--mmproj", help = "Path to the GGUF multimodal projector file")
    private val images by option("--image", help = "Path to an image file. Can be used multiple times.").multiple()
    private val threads by option("-t", "--threads", help = "Number of threads to use").int().default(6)
    private val temperature by option("--temperature", help = "Temperature for sampling").float().default(0.7f)
    private val topP by option("--top-p", help = "Top-p for sampling").float().default(0.9f)
    private val topK by option("--top-k", help = "Top-k for sampling").int().default(40)
    private val minP by option("--min-p", help = "Min-p for sampling").float().default(0.05f)
    private val repeatPenalty by option("--repeat-penalty", help = "Repeat penalty").float().default(1.1f)
    private val repeatLastN by option("--repeat-last-n", help = "Last n tokens to apply repeat penalty").int()
        .default(64)
    private val maxTokens by option("--max-tokens", help = "Maximum tokens to generate").int().default(-1)
    private val validateOnly by option(
        "--validate-model",
        help = "Validate model file without full initialization"
    ).flag(default = false)

    override fun run() {
        if (validateOnly) {
            runValidation()
        } else {
            runGeneration()
        }
    }

    private fun runValidation() {
        echo("Validating model: $model")
        when (val validation = KLlama.validateModel(model)) {
            is KLlamaResult.Success -> {
                val info = validation.value
                echo("Model validation successful!")
                echo("  Name: ${info.name}")
                echo("  Parameters: ${info.parameterCount}")
                echo("  Context size: ${info.contextSize}")
            }

            is KLlamaResult.Error -> echo("Model validation failed: ${validation.message}", err = true)
        }

        mmproj?.let {
            echo("Validating multimodal projector: $it")
            when (val validation = KLlama.validateMmproj(it)) {
                is KLlamaResult.Success -> echo("Multimodal projector validation successful!")
                is KLlamaResult.Error -> echo(
                    "Multimodal projector validation failed: ${validation.message}",
                    err = true
                )
            }
        }
    }

    private fun runGeneration() {
        if (prompt == null) {
            echo("Error: --prompt is required for generation.", err = true)
        }
        if (images.isNotEmpty() && mmproj == null) {
            echo("Error: --image requires --mmproj to be specified.", err = true)
        }

        val samplingParams = SamplingParams(
            temperature,
            topP,
            topK,
            minP,
            1.0f,
            repeatPenalty,
            repeatLastN,
            0.0f,
            0.0f,
            maxTokens
        )
        val sessionParams = SessionParams(
            model,
            mmproj ?: "",
            threads = threads,
            sampling = samplingParams
        )

        echo("Initializing KLlama with model: ${sessionParams.modelPath}")
        if (sessionParams.mmprojPath.isNotEmpty()) {
            echo("Using multimodal projector: ${sessionParams.mmprojPath}")
        }

        val kllama = KLlama()

        val initResult = kllama.initialize(sessionParams, progressCallback = { progress, stage ->
            echo("Progress: ${(progress * 100).toInt()}% - $stage")
        })

        if (initResult is KLlamaResult.Error) {
            echo("KLlama initialization failed: ${initResult.message}", err = true)
        }

        val userImages = images.mapNotNull { imagePath ->
            try {
                echo("Loading image: $imagePath")
                ImageData(File(imagePath).readBytes())
            } catch (e: Exception) {
                echo("Failed to load image $imagePath: ${e.message}", err = true)
                null
            }
        }

        val conversation = listOf(MultimodalMessage(MessageRole.User, prompt!!, userImages))

        echo("\n--- Conversation ---")
        print("User: $prompt")
        if (images.isNotEmpty()) print(" [with ${images.size} image(s)]")
        echo()
        print("Assistant: ")

        val responseResult = kllama.generateResponse(
            conversation = conversation,
            tokenCallback = { token ->
                print(token)
                System.out.flush()
            }
        )
        echo() // Newline after generation finishes

        when (responseResult) {
            is KLlamaResult.Success -> {
                echo("\n--- End of Response ---")
                echo("Generation completed successfully.")
            }

            is KLlamaResult.Error -> {
                echo("\nGeneration failed: ${responseResult.message}", err = true)
            }
        }

        (kllama.getGenerationStats() as? KLlamaResult.Success)?.value?.let { stats ->
            echo("\n--- Stats ---")
            echo("Tokens generated: ${stats.tokensGenerated}")
            echo("Time elapsed: %.2f seconds".format(stats.timeElapsed))
            echo("Tokens per second: ${stats.tokensPerSecond}")
        }
    }
}

fun main(args: Array<String>) = DemoCommand().main(args)