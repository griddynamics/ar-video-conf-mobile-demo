package com.griddynamics.video.conf

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.graphics.Color
import androidx.core.graphics.blue
import androidx.core.graphics.get
import androidx.core.graphics.green
import androidx.core.graphics.red
import com.griddynamics.video.conf.utils.ImageUtils
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ImageSegmentation(
    private val interpreter: Interpreter
) {

    private val segmentationMask: ByteBuffer = ByteBuffer.allocateDirect(imageSize * imageSize * 4)
    var backgroundImage: Bitmap? = null

    init {
        segmentationMask.order(ByteOrder.nativeOrder())
    }

    companion object {
        private const val imageSize = 32
        private const val IMAGE_MEAN = 0.0f
        private const val IMAGE_STD = 255.0f
    }

    @SuppressLint("NewApi")
    fun execute(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        if(backgroundImage == null)
            return bitmap
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
        interpreter.run(normalized, segmentationMask)
        return convertByteBufferMaskToBitmap(
            segmentationMask,
            imageSize,
            imageSize
        )
    }

    private fun convertByteBufferMaskToBitmap(
        inputBuffer: ByteBuffer,
        imageWidth: Int,
        imageHeight: Int
    ): Bitmap {
        val conf = Bitmap.Config.ARGB_8888
        val maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
        inputBuffer.rewind()
        backgroundImage?.let {
            for (y in 0 until imageHeight) {
                for (x in 0 until imageWidth) {
                    val index = y * imageWidth + x
                    val value = inputBuffer.getFloat(index * 4)
                    val segmentColor = Color.argb(
                        ((1 - value) * 255).toInt(),
                        it[x, y].red,
                        it[x, y].green,
                        it[x, y].blue
                    )

                    maskBitmap.setPixel(x, y, segmentColor)
                }
            }
        }
        return maskBitmap
    }
}