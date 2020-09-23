package com.griddynamics.video.conf

import android.annotation.SuppressLint
import android.graphics.*
import android.util.Log
import androidx.core.graphics.scale
import com.commit451.nativestackblur.NativeStackBlur
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import com.griddynamics.video.conf.pref.DeviceInfo
import com.griddynamics.video.conf.utils.ImageUtils
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder


class ImageSegmentation(
    private val interpreter: Interpreter
) {
    private val db = Firebase.firestore

    private val segmentationMask: ByteBuffer =
        ByteBuffer.allocateDirect(IMAGE_SIZE * IMAGE_SIZE * 4)
    var applyBlur = false

    init {
        segmentationMask.order(ByteOrder.nativeOrder())
    }

    companion object {
        private const val BLUR_RADIUS = 25
        private const val IMAGE_SIZE = 32
        private const val IMAGE_MEAN = 0.0f
        private const val IMAGE_STD = 255.0f
    }

    @SuppressLint("NewApi")
    fun execute(bitmap: Bitmap): Bitmap {
        val startTotal = System.currentTimeMillis();
        val scaledBitmap =
            ImageUtils.scaleBitmapAndKeepRatio(
                bitmap,
                IMAGE_SIZE,
                IMAGE_SIZE
            )

        val normalized =
            ImageUtils.bitmapToByteBuffer(
                scaledBitmap,
                IMAGE_SIZE,
                IMAGE_SIZE,
                IMAGE_MEAN,
                IMAGE_STD
            )
        val start = System.currentTimeMillis()
        interpreter.run(normalized, segmentationMask)
        val inference = System.currentTimeMillis() - start
        Log.d("timelap interpreter", (inference).toString())
        val result = convertByteBufferMaskToBitmap(
            bitmap,
            segmentationMask,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        val logTotal = hashMapOf(
            "uuid" to DeviceInfo.uuid,
            "model" to DeviceInfo.model,
            "tag" to "total processing",
            "info" to System.currentTimeMillis() - startTotal
        )
        val logInference = hashMapOf(
            "uuid" to DeviceInfo.uuid,
            "model" to DeviceInfo.model,
            "tag" to "inference",
            "info" to inference
        )
        db.collection("logs").add(logInference)
        db.collection("logs").add(logTotal)

        return result
    }

    private fun convertByteBufferMaskToBitmap(
        bitmapOrig: Bitmap,
        inputBuffer: ByteBuffer,
        imageWidth: Int,
        imageHeight: Int
    ): Bitmap {
        val conf = Bitmap.Config.ARGB_8888
        var maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
        inputBuffer.rewind()
        var start = System.currentTimeMillis()

        for (y in 0 until imageHeight) {
            for (x in 0 until imageWidth) {
                val index = y * imageWidth + x
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
        val origBitmap: Bitmap = bitmapOrig.scale(256, 256)
        maskBitmap = maskBitmap.scale(256, 256)
        Log.d("timelap scale", (System.currentTimeMillis() - start).toString())

        start = System.currentTimeMillis()
        val result = overlay(maskBitmap, origBitmap)
        Log.d("timelap for overlay", (System.currentTimeMillis() - start).toString())

        if (applyBlur) {
            start = System.currentTimeMillis()

            val blur = NativeStackBlur.process(origBitmap, BLUR_RADIUS)
            val resultBlur = overlaySimple(blur, result)
            Log.d("timelap for blur", (System.currentTimeMillis() - start).toString())

            return resultBlur
        }
        return result
    }

    private fun overlay(bmp2: Bitmap, bmp3: Bitmap): Bitmap {
        val bmOverlay = Bitmap.createBitmap(bmp2.width, bmp2.height, bmp2.config)
        val canvas = Canvas(bmOverlay)
        val paint = Paint();
        paint.xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC_ATOP);
        canvas.drawBitmap(bmp2, Matrix(), null)
        canvas.drawBitmap(bmp3, 0f, 0f, paint)
        return bmOverlay
    }

    private fun overlaySimple(bmp1: Bitmap, bmp2: Bitmap): Bitmap {
        val bmOverlay = Bitmap.createBitmap(bmp1.width, bmp1.height, bmp1.config)
        val canvas = Canvas(bmOverlay)
        canvas.drawBitmap(bmp1, Matrix(), null)
        canvas.drawBitmap(bmp2, 0f, 0f, null)
        return bmOverlay
    }
}
