#include <jni.h>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "KLlama.h"
#include "logging/logging.h"

#define LOG_TAG "KLlamaJNI"

// A struct to hold the JVM pointer, which is valid across threads.
struct JniContext {
    JavaVM *jvm = nullptr;
};

static JniContext g_jni_context;

// Attaches the current thread to the JVM and returns a JNIEnv*.
// Must be matched with a call to detach_thread().
static JNIEnv *attach_thread() {
    JNIEnv *env = nullptr;
    if (g_jni_context.jvm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) == JNI_EDETACHED) {
        if (g_jni_context.jvm->AttachCurrentThread(reinterpret_cast<void **>(&env), nullptr) != 0) {
            LOG_ERROR(LOG_TAG, "Failed to attach current thread to JVM");
            return nullptr;
        }
    }
    return env;
}

// Detaches the current thread from the JVM.
static void detach_thread() {
    if (g_jni_context.jvm) {
        // Check if attached before detaching to avoid errors
        void *env_ptr;
        if (g_jni_context.jvm->GetEnv(&env_ptr, JNI_VERSION_1_6) != JNI_EDETACHED) {
            g_jni_context.jvm->DetachCurrentThread();
        }
    }
}

static KLlama *get_handle(JNIEnv *env, jobject thiz) {
    const auto cls = env->GetObjectClass(thiz);
    const auto fid = env->GetFieldID(cls, "nativeHandle", "J");
    return reinterpret_cast<KLlama *>(env->GetLongField(thiz, fid));
}

static void set_handle(JNIEnv *env, jobject thiz, KLlama *kllama) {
    const auto cls = env->GetObjectClass(thiz);
    const auto fid = env->GetFieldID(cls, "nativeHandle", "J");
    env->SetLongField(thiz, fid, reinterpret_cast<jlong>(kllama));
}

class JniString {
public:
    JniString(JNIEnv *env, jstring jstr) : env_(env), jstr_(jstr), cstr_(nullptr) {
        if (jstr) {
            cstr_ = env->GetStringUTFChars(jstr, nullptr);
        }
    }

    ~JniString() {
        if (cstr_) {
            env_->ReleaseStringUTFChars(jstr_, cstr_);
        }
    }

    [[nodiscard]] const char *get() const { return cstr_; }
    [[nodiscard]] std::string str() const { return cstr_ ? std::string(cstr_) : ""; }

private:
    JNIEnv *env_;
    jstring jstr_;
    const char *cstr_;
};

template<typename T>
jobject to_java_result(JNIEnv *env, const KLlamaResult<T> &result,
                       const std::function<jobject(const T &)> &value_converter);

jobject to_java_result(JNIEnv *env, const KLlamaResult<void> &result);

jobject to_java_model_info(JNIEnv *env, const ModelInfo &info);

jobject to_java_memory_info(JNIEnv *env, const MemoryInfo &info);

jobject to_java_generation_stats(JNIEnv *env, const GenerationStats &stats);

jobject to_java_sampling_params(JNIEnv *env, const SamplingParams &params);

SamplingParams from_java_sampling_params(JNIEnv *env, jobject j_params);

SessionParams from_java_session_params(JNIEnv *env, jobject j_params);

std::vector<MultimodalMessage> from_java_multimodal_message_array(JNIEnv *env, jobjectArray j_conversation);

struct JniCallback {
    jobject callback_ref = nullptr;
    jmethodID invoke_method = nullptr;

    JniCallback(JNIEnv *env, jobject callback, const char *class_name, const char *method_signature) {
        if (callback) {
            callback_ref = env->NewGlobalRef(callback);
            if (const auto cls = env->FindClass(class_name)) {
                invoke_method = env->GetMethodID(cls, "invoke", method_signature);
                if (!invoke_method) {
                    LOG_ERROR(LOG_TAG, "Could not find invoke method with signature %s on class %s", method_signature,
                              class_name);
                }
            } else {
                LOG_ERROR(LOG_TAG, "Could not find class: %s", class_name);
            }
        }
    }

    ~JniCallback() {
        if (callback_ref) {
            if (JNIEnv *env = attach_thread()) {
                env->DeleteGlobalRef(callback_ref);
                detach_thread();
            }
        }
    }

    // Make it non-copyable but movable
    JniCallback(const JniCallback &) = delete;

    JniCallback &operator=(const JniCallback &) = delete;

    JniCallback(JniCallback &&other) noexcept : callback_ref(other.callback_ref), invoke_method(other.invoke_method) {
        other.callback_ref = nullptr;
    }

    JniCallback &operator=(JniCallback &&other) noexcept {
        if (this != &other) {
            if (callback_ref) {
                if (JNIEnv *env = attach_thread()) {
                    env->DeleteGlobalRef(callback_ref);
                    detach_thread();
                }
            }
            callback_ref = other.callback_ref;
            invoke_method = other.invoke_method;
            other.callback_ref = nullptr;
        }
        return *this;
    }
};

// Wraps a Kotlin CancellationToken jobject for use in C++.
class JniCancellationToken : public CancellationToken {
public:
    JniCancellationToken(JNIEnv *env, jobject j_token) {
        if (!j_token) return;
        j_token_ref = env->NewGlobalRef(j_token);
        const auto cls = env->GetObjectClass(j_token_ref);
        is_cancelled_method = env->GetMethodID(cls, "isCancelled", "()Z");
    }

    ~JniCancellationToken() {
        if (j_token_ref) {
            if (JNIEnv *env = attach_thread()) {
                env->DeleteGlobalRef(j_token_ref);
                detach_thread();
            }
        }
    }

    // FIX: Removed 'override' as the base CancellationToken::isCancelled is not virtual.
    [[nodiscard]] bool isCancelled() const {
        if (!j_token_ref || !is_cancelled_method) return false;
        JNIEnv *env = attach_thread();
        if (!env) return false;
        const auto result = env->CallBooleanMethod(j_token_ref, is_cancelled_method);
        detach_thread();
        return result;
    }

private:
    jobject j_token_ref = nullptr;
    jmethodID is_cancelled_method = nullptr;
};

// Wraps a Kotlin ProgressCallback jobject.
ProgressCallback create_progress_callback(JNIEnv *env, jobject j_callback) {
    if (!j_callback) return nullptr;
    auto cb_wrapper = std::make_shared<JniCallback>(env, j_callback, "kotlin/jvm/functions/Function2",
                                                    "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

    return [cb_wrapper](float progress, const std::string &stage) {
        if (!cb_wrapper || !cb_wrapper->callback_ref || !cb_wrapper->invoke_method) return;

        JNIEnv *env = attach_thread();
        if (!env) return;

        // Box the float primitive into a java.lang.Float object
        const auto float_cls = env->FindClass("java/lang/Float");
        const auto float_ctor = env->GetMethodID(float_cls, "<init>", "(F)V");
        const auto j_progress = env->NewObject(float_cls, float_ctor, progress);

        const auto j_stage = env->NewStringUTF(stage.c_str());

        // Call the invoke method
        const auto result = env->CallObjectMethod(cb_wrapper->callback_ref, cb_wrapper->invoke_method, j_progress,
                                                  j_stage);
        if (result) env->DeleteLocalRef(result); // Clean up the returned Unit object

        env->DeleteLocalRef(j_stage);
        env->DeleteLocalRef(j_progress);
        env->DeleteLocalRef(float_cls);
        detach_thread();
    };
}

// Wraps a Kotlin TokenCallback jobject.
TokenCallback create_token_callback(JNIEnv *env, jobject j_callback) {
    if (!j_callback) return nullptr;
    // Use the JniCallback RAII wrapper and correct method signature for Function1
    auto cb_wrapper = std::make_shared<JniCallback>(env, j_callback, "kotlin/jvm/functions/Function1",
                                                    "(Ljava/lang/Object;)Ljava/lang/Object;");

    return [cb_wrapper](const std::string &token) {
        if (!cb_wrapper || !cb_wrapper->callback_ref || !cb_wrapper->invoke_method) return;

        JNIEnv *env = attach_thread();
        if (!env) return;

        const auto j_token = env->NewStringUTF(token.c_str());
        if (const auto result = env->CallObjectMethod(cb_wrapper->callback_ref, cb_wrapper->invoke_method, j_token)) {
            env->DeleteLocalRef(result); // Clean up the returned Unit object
        }

        env->DeleteLocalRef(j_token);
        detach_thread();
    };
}

extern "C" JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, [[maybe_unused]] void *reserved) {
    g_jni_context.jvm = vm;
    return JNI_VERSION_1_6;
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_actinis_kllama_1cpp_KLlama_00024Companion_validateModelNative(
    JNIEnv *env,
    jobject,
    jstring j_model_path
) {
    const JniString model_path(env, j_model_path);
    const auto result = KLlama::validateModel(model_path.str());

    return to_java_result<ModelInfo>(env, result, std::function([&](const ModelInfo &info) {
        return to_java_model_info(env, info);
    }));
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_actinis_kllama_1cpp_KLlama_00024Companion_validateMmprojNative(
    JNIEnv *env,
    jobject,
    jstring j_mmproj_path
) {
    const JniString mmproj_path(env, j_mmproj_path);
    const auto result = KLlama::validateMmproj(mmproj_path.str());
    return to_java_result(env, result);
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_actinis_kllama_1cpp_KLlama_00024Companion_validateImageDataNative(
    JNIEnv *env,
    jobject,
    jobject j_image_data
) {
    const auto image_data_cls = env->FindClass("io/actinis/kllama_cpp/data/model/message/ImageData");
    const auto get_data = env->GetMethodID(image_data_cls, "getData", "()[B");
    const auto j_data = reinterpret_cast<jbyteArray>(env->CallObjectMethod(j_image_data, get_data));

    auto *data_ptr = env->GetByteArrayElements(j_data, nullptr);
    const auto len = env->GetArrayLength(j_data);
    ImageData image_data;
    image_data.data.assign(reinterpret_cast<uint8_t *>(data_ptr), reinterpret_cast<uint8_t *>(data_ptr) + len);
    env->ReleaseByteArrayElements(j_data, data_ptr, JNI_ABORT);

    const auto result = KLlama::validateImageData(image_data);

    return to_java_result<std::vector<uint8_t> >(env, result, std::function(
                                                     [&](const std::vector<uint8_t> &data) {
                                                         const auto j_byte_array = env->NewByteArray(
                                                             static_cast<jsize>(data.size())
                                                         );
                                                         env->SetByteArrayRegion(
                                                             j_byte_array,
                                                             0,
                                                             static_cast<jsize>(data.size()),
                                                             reinterpret_cast<const jbyte *>(data.data()));
                                                         return static_cast<jobject>(j_byte_array);
                                                     }));
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_actinis_kllama_1cpp_KLlama_initializeNative(
    JNIEnv *env,
    jobject thiz,
    jobject j_params,
    jobject j_progress_cb,
    jobject j_cancel_token
) {
    const auto kllama = new KLlama();

    const auto params = from_java_session_params(env, j_params);
    const auto progress_callback = create_progress_callback(env, j_progress_cb);
    const auto cancellation_token = std::make_unique<JniCancellationToken>(env, j_cancel_token);

    const auto result = kllama->initialize(params, progress_callback, cancellation_token.get());

    if (result.isSuccess()) {
        set_handle(env, thiz, kllama);
    } else {
        delete kllama;
        set_handle(env, thiz, nullptr);
    }

    return to_java_result(env, result);
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_actinis_kllama_1cpp_KLlama_generateResponseNative(
    JNIEnv *env,
    jobject thiz,
    jobjectArray j_conversation,
    jobject j_sampling,
    jobject j_token_cb,
    jobject j_progress_cb,
    jobject j_cancel_token
) {
    KLlama *kllama = get_handle(env, thiz);
    if (!kllama) {
        const KLlamaResult<std::string> not_init_res(KLlamaError::NotInitialized, "KLlama not initialized");
        return to_java_result<std::string>(env, not_init_res, nullptr);
    }

    const auto conversation = from_java_multimodal_message_array(env, j_conversation);
    const auto sampling = from_java_sampling_params(env, j_sampling);
    const auto token_callback = create_token_callback(env, j_token_cb);
    const auto progress_callback = create_progress_callback(env, j_progress_cb);
    const auto cancellation_token = std::make_unique<JniCancellationToken>(env, j_cancel_token);

    const auto result = kllama->generateResponse(conversation, sampling, token_callback, progress_callback,
                                                 cancellation_token.get());

    return to_java_result<std::string>(env, result, std::function<jobject(const std::string &)>(
                                           [&](const std::string &value) {
                                               return env->NewStringUTF(value.c_str());
                                           }));
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_actinis_kllama_1cpp_KLlama_getModelInfoNative(JNIEnv *env, jobject thiz) {
    const KLlama *kllama = get_handle(env, thiz);
    if (!kllama) {
        const KLlamaResult<ModelInfo> not_init_res(KLlamaError::NotInitialized, "KLlama not initialized");
        return to_java_result<ModelInfo>(env, not_init_res, nullptr);
    }
    const auto result = kllama->getModelInfo();
    return to_java_result<ModelInfo>(env, result, std::function([&](const ModelInfo &info) {
        return to_java_model_info(env, info);
    }));
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_actinis_kllama_1cpp_KLlama_getMemoryInfoNative(JNIEnv *env, jobject thiz) {
    const KLlama *kllama = get_handle(env, thiz);
    if (!kllama) {
        const KLlamaResult<MemoryInfo> not_init_res(KLlamaError::NotInitialized, "KLlama not initialized");
        return to_java_result<MemoryInfo>(env, not_init_res, nullptr);
    }
    const auto result = kllama->getMemoryInfo();
    return to_java_result<MemoryInfo>(env, result, std::function(
                                          [&](const MemoryInfo &info) {
                                              return to_java_memory_info(env, info);
                                          }));
}

extern "C" JNIEXPORT jobject JNICALL
Java_io_actinis_kllama_1cpp_KLlama_getGenerationStatsNative(JNIEnv *env, jobject thiz) {
    const KLlama *kllama = get_handle(env, thiz);
    if (!kllama) {
        const KLlamaResult<GenerationStats> not_init_res(KLlamaError::NotInitialized, "KLlama not initialized");
        return to_java_result<GenerationStats>(env, not_init_res, nullptr);
    }
    const auto result = kllama->getGenerationStats();
    return to_java_result<GenerationStats>(env, result, std::function(
                                               [&](const GenerationStats &stats) {
                                                   return to_java_generation_stats(env, stats);
                                               }));
}

extern "C" JNIEXPORT void JNICALL
Java_io_actinis_kllama_1cpp_KLlama_freeMemory(JNIEnv *env, jobject thiz) {
    if (const KLlama *kllama = get_handle(env, thiz)) {
        delete kllama;
        set_handle(env, thiz, nullptr);
        LOG_INFO(LOG_TAG, "Native KLlama instance freed.");
    }
}

// Generic result converter
template<typename T>
jobject to_java_result(JNIEnv *env, const KLlamaResult<T> &result,
                       const std::function<jobject(const T &)> &value_converter) {
    if (result.isSuccess()) {
        const auto success_cls = env->FindClass("io/actinis/kllama_cpp/data/model/result/KLlamaResult$Success");
        const auto ctor = env->GetMethodID(success_cls, "<init>", "(Ljava/lang/Object;)V");
        const auto value_obj = value_converter ? value_converter(result.value) : nullptr;
        const auto new_obj = env->NewObject(success_cls, ctor, value_obj);
        if (value_obj) env->DeleteLocalRef(value_obj);
        return new_obj;
    } else {
        const auto error_cls = env->FindClass("io/actinis/kllama_cpp/data/model/result/KLlamaResult$Error");
        const auto ctor = env->GetMethodID(error_cls, "<init>",
                                           "(Lio/actinis/kllama_cpp/data/model/result/KLlamaError;Ljava/lang/String;)V");

        const auto error_enum_cls = env->FindClass("io/actinis/kllama_cpp/data/model/result/KLlamaError");
        const auto error_field = env->GetStaticFieldID(error_enum_cls, KLlama::errorToString(result.error).c_str(),
                                                       "Lio/actinis/kllama_cpp/data/model/result/KLlamaError;");
        const auto error_enum_val = env->GetStaticObjectField(error_enum_cls, error_field);

        const auto msg = env->NewStringUTF(result.errorMessage.c_str());
        const auto new_obj = env->NewObject(error_cls, ctor, error_enum_val, msg);
        env->DeleteLocalRef(msg);
        env->DeleteLocalRef(error_enum_val);
        return new_obj;
    }
}

// Specialization for KLlamaResult<void>
jobject to_java_result(JNIEnv *env, const KLlamaResult<void> &result) {
    if (result.isSuccess()) {
        // For void success, return a Success(Unit)
        const auto success_cls = env->FindClass("io/actinis/kllama_cpp/data/model/result/KLlamaResult$Success");
        const auto ctor = env->GetMethodID(success_cls, "<init>", "(Ljava/lang/Object;)V");
        const auto unit_cls = env->FindClass("kotlin/Unit");
        const auto instance_field = env->GetStaticFieldID(unit_cls, "INSTANCE", "Lkotlin/Unit;");
        const auto unit_obj = env->GetStaticObjectField(unit_cls, instance_field);
        const auto new_obj = env->NewObject(success_cls, ctor, unit_obj);
        env->DeleteLocalRef(unit_obj);
        return new_obj;
    }
    // For error, use the generic implementation
    return to_java_result<std::nullptr_t>(env, KLlamaResult<std::nullptr_t>(result.error, result.errorMessage),
                                          nullptr);
}

// C++ to Java Converters

jobject to_java_model_info(JNIEnv *env, const ModelInfo &info) {
    const auto cls = env->FindClass("io/actinis/kllama_cpp/data/model/info/ModelInfo");
    const auto ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;Ljava/lang/String;JIZLjava/util/List;)V");

    const auto name = env->NewStringUTF(info.name.c_str());
    const auto arch = env->NewStringUTF(info.architecture.c_str());

    const auto list_cls = env->FindClass("java/util/ArrayList");
    const auto list_ctor = env->GetMethodID(list_cls, "<init>", "()V");
    const auto list_add = env->GetMethodID(list_cls, "add", "(Ljava/lang/Object;)Z");
    const auto capabilities = env->NewObject(list_cls, list_ctor);
    for (const auto &cap: info.capabilities) {
        const auto j_cap = env->NewStringUTF(cap.c_str());
        env->CallBooleanMethod(capabilities, list_add, j_cap);
        env->DeleteLocalRef(j_cap);
    }

    const auto new_obj = env->NewObject(cls, ctor, name, arch, info.parameterCount, info.contextSize,
                                        info.supportsVision, capabilities);
    env->DeleteLocalRef(name);
    env->DeleteLocalRef(arch);
    env->DeleteLocalRef(capabilities);
    return new_obj;
}

jobject to_java_memory_info(JNIEnv *env, const MemoryInfo &info) {
    const auto cls = env->FindClass("io/actinis/kllama_cpp/data/model/info/MemoryInfo");
    const auto ctor = env->GetMethodID(cls, "<init>", "(JJJJ)V");
    return env->NewObject(cls, ctor, static_cast<jlong>(info.modelMemoryMB), static_cast<jlong>(info.contextMemoryMB),
                          static_cast<jlong>(info.totalMemoryMB), static_cast<jlong>(info.availableMemoryMB));
}

jobject to_java_generation_stats(JNIEnv *env, const GenerationStats &stats) {
    const auto cls = env->FindClass("io/actinis/kllama_cpp/data/model/info/GenerationStats");
    const auto ctor = env->GetMethodID(cls, "<init>",
                                       "(IIFLio/actinis/kllama_cpp/data/model/GenerationState;Lio/actinis/kllama_cpp/data/model/params/SamplingParams;)V");

    const auto state_enum_cls = env->FindClass("io/actinis/kllama_cpp/data/model/GenerationState");
    const char *state_name;
    switch (stats.state) {
        case GenerationState::Idle: state_name = "Idle";
            break;
        case GenerationState::Initializing: state_name = "Initializing";
            break;
        case GenerationState::TokenizingPrompt: state_name = "TokenizingPrompt";
            break;
        case GenerationState::ProcessingImages: state_name = "ProcessingImages";
            break;
        case GenerationState::Generating: state_name = "Generating";
            break;
        case GenerationState::Finished: state_name = "Finished";
            break;
        case GenerationState::Cancelled: state_name = "Cancelled";
            break;
        case GenerationState::Error: state_name = "Error";
            break;
        default: state_name = "Error";
            break;
    }
    const auto state_field = env->GetStaticFieldID(state_enum_cls, state_name,
                                                   "Lio/actinis/kllama_cpp/data/model/GenerationState;");
    const auto state_enum_val = env->GetStaticObjectField(state_enum_cls, state_field);

    const auto sampling_params = to_java_sampling_params(env, stats.sampling);

    const auto new_obj = env->NewObject(cls, ctor, stats.tokensGenerated, stats.tokensPerSecond, stats.timeElapsed,
                                        state_enum_val, sampling_params);
    env->DeleteLocalRef(state_enum_val);
    env->DeleteLocalRef(sampling_params);
    return new_obj;
}


jobject to_java_sampling_params(JNIEnv *env, const SamplingParams &params) {
    const auto cls = env->FindClass("io/actinis/kllama_cpp/data/model/params/SamplingParams");
    const auto ctor = env->GetMethodID(cls, "<init>", "(FFIFFIFFFI)V");
    return env->NewObject(cls, ctor, params.temperature, params.topP, params.topK, params.minP, params.typicalP,
                          params.repeatPenalty, params.repeatLastN, params.frequencyPenalty, params.presencePenalty,
                          params.nPredict);
}

// Java to C++ Converters

#define GET_FIELD(env, cls, obj, name, sig, jni_type) env->Get##jni_type##Field(obj, env->GetFieldID(cls, name, sig))
#define GET_STRING_FIELD(env, cls, obj, name) (jstring)env->GetObjectField(obj, env->GetFieldID(cls, name, "Ljava/lang/String;"))
#define GET_OBJ_FIELD(env, cls, obj, name, type) env->GetObjectField(obj, env->GetFieldID(cls, name, "L" type ";"))

SamplingParams from_java_sampling_params(JNIEnv *env, jobject j_params) {
    const auto cls = env->GetObjectClass(j_params);
    SamplingParams p;
    p.temperature = GET_FIELD(env, cls, j_params, "temperature", "F", Float);
    p.topP = GET_FIELD(env, cls, j_params, "topP", "F", Float);
    p.topK = GET_FIELD(env, cls, j_params, "topK", "I", Int);
    p.minP = GET_FIELD(env, cls, j_params, "minP", "F", Float);
    p.typicalP = GET_FIELD(env, cls, j_params, "typicalP", "F", Float);
    p.repeatPenalty = GET_FIELD(env, cls, j_params, "repeatPenalty", "F", Float);
    p.repeatLastN = GET_FIELD(env, cls, j_params, "repeatLastN", "I", Int);
    p.frequencyPenalty = GET_FIELD(env, cls, j_params, "frequencyPenalty", "F", Float);
    p.presencePenalty = GET_FIELD(env, cls, j_params, "presencePenalty", "F", Float);
    p.nPredict = GET_FIELD(env, cls, j_params, "nPredict", "I", Int);
    return p;
}

SessionParams from_java_session_params(JNIEnv *env, jobject j_params) {
    const auto cls = env->GetObjectClass(j_params);
    SessionParams p;
    p.modelPath = JniString(env, GET_STRING_FIELD(env, cls, j_params, "modelPath")).str();
    p.mmprojPath = JniString(env, GET_STRING_FIELD(env, cls, j_params, "mmprojPath")).str();
    p.contextSize = GET_FIELD(env, cls, j_params, "contextSize", "I", Int);
    p.batch = GET_FIELD(env, cls, j_params, "batch", "I", Int);
    p.gpuLayers = GET_FIELD(env, cls, j_params, "gpuLayers", "I", Int);
    p.mmprojUseGpu = GET_FIELD(env, cls, j_params, "mmprojUseGpu", "Z", Boolean);
    p.threads = GET_FIELD(env, cls, j_params, "threads", "I", Int);
    p.verbosity = GET_FIELD(env, cls, j_params, "verbosity", "I", Int);

    const auto j_sampling = GET_OBJ_FIELD(env, cls, j_params, "sampling",
                                          "io/actinis/kllama_cpp/data/model/params/SamplingParams");
    p.sampling = from_java_sampling_params(env, j_sampling);
    env->DeleteLocalRef(j_sampling);
    return p;
}

std::vector<MultimodalMessage> from_java_multimodal_message_array(JNIEnv *env, jobjectArray j_conversation) {
    std::vector<MultimodalMessage> conversation;
    const auto count = env->GetArrayLength(j_conversation);
    conversation.reserve(count);

    const auto msg_cls = env->FindClass("io/actinis/kllama_cpp/data/model/message/MultimodalMessage");
    const auto get_role = env->GetMethodID(msg_cls, "getRole",
                                           "()Lio/actinis/kllama_cpp/data/model/message/MessageRole;");
    const auto get_content = env->GetMethodID(msg_cls, "getContent", "()Ljava/lang/String;");
    const auto get_images = env->GetMethodID(msg_cls, "getImages", "()Ljava/util/List;");

    const auto role_cls = env->FindClass("io/actinis/kllama_cpp/data/model/message/MessageRole");
    const auto role_ordinal = env->GetMethodID(role_cls, "ordinal", "()I");

    const auto list_cls = env->FindClass("java/util/List");
    const auto list_size = env->GetMethodID(list_cls, "size", "()I");
    const auto list_get = env->GetMethodID(list_cls, "get", "(I)Ljava/lang/Object;");

    const auto image_data_cls = env->FindClass("io/actinis/kllama_cpp/data/model/message/ImageData");
    const auto get_data = env->GetMethodID(image_data_cls, "getData", "()[B");

    for (jsize i = 0; i < count; ++i) {
        const auto j_msg = env->GetObjectArrayElement(j_conversation, i);
        MultimodalMessage msg;

        // Role
        const auto j_role = env->CallObjectMethod(j_msg, get_role);
        int role_idx = env->CallIntMethod(j_role, role_ordinal);
        msg.role = static_cast<MessageRole>(role_idx);
        env->DeleteLocalRef(j_role);

        // Content
        const auto j_content = reinterpret_cast<jstring>(env->CallObjectMethod(j_msg, get_content));
        msg.content = JniString(env, j_content).str();
        env->DeleteLocalRef(j_content);

        // Images
        const auto j_images_list = env->CallObjectMethod(j_msg, get_images);
        const auto images_count = env->CallIntMethod(j_images_list, list_size);
        for (jint j = 0; j < images_count; ++j) {
            const auto j_image_data = env->CallObjectMethod(j_images_list, list_get, j);
            const auto j_data_array = reinterpret_cast<jbyteArray>(env->CallObjectMethod(j_image_data, get_data));

            auto *data_ptr = env->GetByteArrayElements(j_data_array, nullptr);
            const auto len = env->GetArrayLength(j_data_array);
            ImageData image_data;
            image_data.data.assign(reinterpret_cast<uint8_t *>(data_ptr), reinterpret_cast<uint8_t *>(data_ptr) + len);
            env->ReleaseByteArrayElements(j_data_array, data_ptr, JNI_ABORT);

            msg.images.push_back(std::move(image_data));

            env->DeleteLocalRef(j_data_array);
            env->DeleteLocalRef(j_image_data);
        }
        env->DeleteLocalRef(j_images_list);

        conversation.push_back(std::move(msg));
        env->DeleteLocalRef(j_msg);
    }
    return conversation;
}
