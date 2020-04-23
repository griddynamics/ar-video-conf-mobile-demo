package com.griddynamics.video.conf.tf

import android.graphics.Bitmap

data class ModelExecutionResult(
    val bitmapResult: Bitmap,
    val bitmapOriginal: Bitmap,
    val bitmapMaskOnly: Bitmap,
    val executionLog: String,
    val itemsFound: Set<Int>
) {
    fun recycle() {
        listOf(bitmapOriginal, bitmapResult, bitmapMaskOnly).forEach { it.recycle() }
    }
}