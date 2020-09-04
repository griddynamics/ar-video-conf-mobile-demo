package com.griddynamics.video.conf


import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.text.TextUtils
import androidx.core.graphics.drawable.toBitmap
import androidx.core.graphics.get
import androidx.core.graphics.scale

import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.*


class DeepLabLite {
    var mModelBuffer: MappedByteBuffer? = null
    private var mImageData: ByteBuffer? = null
    private var mOutputs: ByteBuffer? = null
    private lateinit var mSegmentBits: Array<IntArray>
    private lateinit var mSegmentColors: IntArray
    private lateinit var bitmapBeach: Bitmap

    fun initialize(context: Context?): Boolean {
        if (context == null) {
            return false
        }
        mModelBuffer = loadModelFile(context, MODEL_PATH)
        if (mModelBuffer == null) {
            return false
        }
        mImageData = ByteBuffer.allocateDirect(
            1 * inputSize * inputSize * COLOR_CHANNELS * BYTES_PER_POINT
        )
        mImageData?.order(ByteOrder.nativeOrder())
        mOutputs =
            ByteBuffer.allocateDirect(1 * inputSize * inputSize * NUM_CLASSES * BYTES_PER_POINT)
        mOutputs?.order(ByteOrder.nativeOrder())
        mSegmentBits = Array(inputSize) {
            IntArray(
                inputSize
            )
        }
        mSegmentColors = IntArray(NUM_CLASSES)
        for (i in 0 until NUM_CLASSES) {
            if (i == 0) {
                mSegmentColors[i] = Color.BLACK
            } else {
                mSegmentColors[i] = Color.TRANSPARENT
            }
        }
        bitmapBeach = context.getDrawable(R.drawable.beach)?.toBitmap()!!.scale(257, 257)

        return mModelBuffer != null
    }

    val isInitialized: Boolean
        get() = mModelBuffer != null

    fun segment(bitmap: Bitmap?): Bitmap? {
        var bitmap = bitmap
        if (mModelBuffer == null) {

            return null
        }
        if (bitmap == null) {
            return null
        }
        var w = bitmap.width
        var h = bitmap.height

        if (w > inputSize || h > inputSize) {
            return null
        }
        if (w < inputSize || h < inputSize) {
            //     bitmap = BitmapUtils.extendBitmap(
            //         bitmap, inputSize, inputSize, Color.BLACK
            //     )
            w = bitmap.width
            h = bitmap.height

        }
        mImageData!!.rewind()
        mOutputs!!.rewind()
        val mIntValues = IntArray(w * h)
        bitmap.getPixels(mIntValues, 0, w, 0, 0, w, h)
        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                if (pixel >= mIntValues.size) {
                    break
                }
                val `val` = mIntValues[pixel++]
                mImageData!!.putFloat(((`val` shr 16 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                mImageData!!.putFloat(((`val` shr 8 and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
                mImageData!!.putFloat(((`val` and 0xFF) - IMAGE_MEAN) / IMAGE_STD)
            }
        }
        val options = Interpreter.Options()
        val interpreter = Interpreter(
            mModelBuffer!!, options
        )
        debugInputs(interpreter)
        debugOutputs(interpreter)
        val start = System.currentTimeMillis()

        interpreter.run(mImageData, mOutputs)
        val end = System.currentTimeMillis()


        val maskBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        fillZeroes(mSegmentBits)
        var maxVal = 0f
        var `val` = 0f
        for (y in 0 until h) {
            for (x in 0 until w) {
                mSegmentBits[x][y] = 0
                for (c in 0 until NUM_CLASSES) {
                    `val` =
                        mOutputs!!.getFloat((y * w * NUM_CLASSES + x * NUM_CLASSES + c) * BYTES_PER_POINT)
                    if (c == 0 || `val` > maxVal) {
                        maxVal = `val`
                        mSegmentBits[x][y] = c
                    }
                }
                if (mSegmentBits[x][y] == 0) {
                    maskBitmap.setPixel(x, y, bitmapBeach[x, y])
                } else {
                    maskBitmap.setPixel(x, y, mSegmentColors[mSegmentBits[x][y]])
                }
            }
        }
        return maskBitmap
    }

    private fun fillZeroes(array: Array<IntArray>?) {
        if (array == null) {
            return
        }
        var r: Int
        r = 0
        while (r < array.size) {
            Arrays.fill(array[r], 0)
            r++
        }
    }

    companion object {
        private const val MODEL_PATH = "deeplabv3_257_mv_gpu.tflite"
        private const val USE_GPU = false
        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_STD = 128.0f
        val inputSize = 257
        private const val NUM_CLASSES = 21
        private const val COLOR_CHANNELS = 3
        private const val BYTES_PER_POINT = 4
        private val RANDOM = Random(System.currentTimeMillis())
        private fun debugInputs(interpreter: Interpreter?) {
            if (interpreter == null) {
                return
            }
            val numOfInputs = interpreter.inputTensorCount

            for (i in 0 until numOfInputs) {
                val t = interpreter.getInputTensor(i)
            }
        }

        private fun debugOutputs(interpreter: Interpreter?) {
            if (interpreter == null) {
                return
            }
            val numOfOutputs = interpreter.outputTensorCount

            for (i in 0 until numOfOutputs) {
                val t = interpreter.getOutputTensor(i)
            }
        }

        private fun loadModelFile(context: Context?, modelFile: String): MappedByteBuffer? {
            if (context == null
                || TextUtils.isEmpty(modelFile)
            ) {
                return null
            }
            var buffer: MappedByteBuffer? = null
            buffer = try {
                val df = context.assets.openFd(modelFile)
                val inputStream = FileInputStream(df.fileDescriptor)
                val fileChannel = inputStream.channel
                val startOffset = df.startOffset
                val declaredLength = df.declaredLength
                fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            } catch (e: IOException) {
                null
            }
            return buffer
        }
    }
}