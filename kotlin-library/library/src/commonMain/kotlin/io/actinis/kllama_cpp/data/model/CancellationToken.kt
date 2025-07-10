@file:OptIn(ExperimentalAtomicApi::class)

package io.actinis.kllama_cpp.data.model

import kotlin.concurrent.atomics.AtomicBoolean
import kotlin.concurrent.atomics.ExperimentalAtomicApi

class CancellationToken(initialValue: Boolean = false) {
    val cancelled = AtomicBoolean(initialValue)
    fun cancel() = cancelled.store(true)
    fun isCancelled(): Boolean = cancelled.load()
    fun reset() = cancelled.store(false)
}
