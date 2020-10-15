package com.griddynamics.video.conf

import android.annotation.SuppressLint
import android.graphics.*
import android.util.Log
import androidx.core.graphics.scale
import com.griddynamics.video.conf.pref.Settings
import com.griddynamics.video.conf.utils.ImageUtils
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder


class ImageSegmentation(
    private val interpreter: Interpreter,
    private val logger: Logger,
    private val imageSize: Int
) {
    private val segmentationMask: ByteBuffer =
        ByteBuffer.allocateDirect(imageSize * imageSize * 4)

    init {
        segmentationMask.order(ByteOrder.nativeOrder())
    }

    companion object {
        private const val IMAGE_MEAN = 0.0f
        private const val IMAGE_STD = 255.0f
    }

    @SuppressLint("NewApi")
    fun execute(bitmap: Bitmap): Pair<Bitmap, Bitmap> {
        val startTotal = System.currentTimeMillis()
        val scaledBitmap =
            ImageUtils.scaleBitmapAndKeepRatio(
                bitmap,
                imageSize,
                imageSize
            )

        val normalized =
            ImageUtils.bitmapToByteBuffer(
                scaledBitmap,
                imageSize,
                imageSize,
                IMAGE_MEAN,
                IMAGE_STD
            )
        val start = System.currentTimeMillis()
        interpreter.run(normalized, segmentationMask)
        val inference = System.currentTimeMillis() - start
        Log.d("timelap interpreter", (inference).toString())
        val result = convertByteBufferMaskToBitmap(
            bitmap,
            segmentationMask
        )

        logger.logInfo("inference", inference)
        logger.logInfo("total processing", System.currentTimeMillis() - startTotal)

        return result
    }

    private fun convertByteBufferMaskToBitmap(
        bitmapOrig: Bitmap,
        inputBuffer: ByteBuffer
    ): Pair<Bitmap, Bitmap> {
        val conf = Bitmap.Config.ARGB_8888
        val maskBitmap = Bitmap.createBitmap(imageSize, imageSize, conf)
        inputBuffer.rewind()
        var start = System.currentTimeMillis()

        if (Settings.modelRound > 0) {
            onModeEnabled(maskBitmap, inputBuffer)
        } else {
            onModeDisabled(maskBitmap, inputBuffer)
        }

        val origBitmap: Bitmap = bitmapOrig.scale(Settings.modelScale, Settings.modelScale)
        val scaledBitmap = maskBitmap.scale(Settings.modelScale, Settings.modelScale)
        Log.d("timelap scale", (System.currentTimeMillis() - start).toString())

        start = System.currentTimeMillis()
        val result = overlay(scaledBitmap, origBitmap)
        Log.d("timelap for overlay", (System.currentTimeMillis() - start).toString())
        return Pair(result, maskBitmap)
    }

    private fun onModeEnabled(maskBitmap: Bitmap, inputBuffer: ByteBuffer) {
        for (y in 0 until imageSize) {
            for (x in 0 until imageSize) {
                val index = y * imageSize + x
                val ibValue = inputBuffer.getFloat(index * 4)
                val value = if (ibValue < Settings.modelRound) 0f else ibValue
                val segmentColor = Color.argb(
                    ((value) * 255).toInt(),
                    0,
                    0,
                    0
                )

                maskBitmap.setPixel(x, y, segmentColor)
            }
        }
    }

    private fun onModeDisabled(maskBitmap: Bitmap, inputBuffer: ByteBuffer) {
        for (y in 0 until imageSize) {
            for (x in 0 until imageSize) {
                val index = y * imageSize + x
                val value = inputBuffer.getFloat(index * 4)
                val segmentColor = Color.argb(
                    ((value) * 255).toInt(),
                    0,
                    0,
                    0
                )

                maskBitmap.setPixel(x, y, segmentColor)
            }
        }
    }

    private fun overlay(bmp2: Bitmap, bmp3: Bitmap): Bitmap {
        val bmOverlay = Bitmap.createBitmap(bmp2.width, bmp2.height, bmp2.config)
        val canvas = Canvas(bmOverlay)
        val paint = Paint()
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_ATOP)
        canvas.drawBitmap(bmp2, Matrix(), null)
        canvas.drawBitmap(bmp3, 0f, 0f, paint)
        return bmOverlay
    }
}
