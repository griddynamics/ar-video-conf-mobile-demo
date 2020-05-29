package com.griddynamics.video.conf.tf

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.ColorUtils
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Class responsible to run the Image Segmentation model.
 * more information about the DeepLab model being used can
 * be found here
 */
class ImageSegmentationModelExecutorCustomStatic(
    private val context: Context,
    private var useGPU: Boolean = false
) {
    private var gpuDelegate: GpuDelegate? = null

    private var segmentationMask: ByteBuffer? = null
    private var interpreter: Interpreter? = null

    private var fullTimeExecutionTime = 0L
    private var preprocessTime = 0L
    private var imageSegmentationTime = 0L
    private var maskFlatteningTime = 0L

    private var numberThreads = 4

    private val executorService: ExecutorService = Executors.newSingleThreadExecutor()

    fun initialize(): Task<Nothing?> {
        return Tasks.call(executorService, Callable {
            interpreter = getInterpreter(context, imageSegmentationModel, useGPU)
            segmentationMask =
                ByteBuffer.allocateDirect(imageSize * imageSize * 4)
            segmentationMask!!.order(ByteOrder.nativeOrder())
            null
        })
    }

    fun executeAsync(bitmap: Bitmap): Task<ModelExecutionResult> {
        return Tasks.call(executorService, Callable { execute(bitmap) })
    }

    /**
     * Use this method with executorService. Otherwise GpuDelegate will cause exception
     */
    @Throws(Exception::class)
    fun execute(bitmap: Bitmap): ModelExecutionResult {
        fullTimeExecutionTime = SystemClock.uptimeMillis()

        preprocessTime = SystemClock.uptimeMillis()
        val scaledBitmap =
            ImageUtils.scaleBitmapAndKeepRatio(
                bitmap,
                imageSize,
                imageSize
            )

        val contentArray =
            ImageUtils.bitmapToByteBuffer(
                scaledBitmap,
                imageSize,
                imageSize,
                IMAGE_MEAN,
                IMAGE_STD
            )
        preprocessTime = SystemClock.uptimeMillis() - preprocessTime

        imageSegmentationTime = SystemClock.uptimeMillis()
        try {
            interpreter!!.run(contentArray, segmentationMask)
        } catch (ex: Exception) {
            Log.e("MODEL_ERROR", ex.toString())
        }

        imageSegmentationTime = SystemClock.uptimeMillis() - imageSegmentationTime
        Log.d(TAG, "Time to run the model $imageSegmentationTime")

        maskFlatteningTime = SystemClock.uptimeMillis()
        val (maskImageApplied, maskOnly, itemsFound) =
            convertBytebufferMaskToBitmap(
                segmentationMask!!,
                imageSize,
                imageSize, scaledBitmap
            )
        maskFlatteningTime = SystemClock.uptimeMillis() - maskFlatteningTime
        Log.d(TAG, "Time to flatten the mask result $maskFlatteningTime")

        fullTimeExecutionTime = SystemClock.uptimeMillis() - fullTimeExecutionTime
        Log.d(TAG, "Total time execution $fullTimeExecutionTime")

        return ModelExecutionResult(
            maskImageApplied,
            scaledBitmap,
            maskOnly,
            formatExecutionLog(),
            itemsFound
        )
    }

    // base: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/demo/app/src/main/java/com/example/android/tflitecamerademo/ImageClassifier.java
    @Throws(IOException::class)
    private fun loadModelFile(context: Context, modelFile: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelFile)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileDescriptor.close()
        return retFile
    }

    @Throws(IOException::class)
    private fun getInterpreter(
        context: Context,
        modelName: String,
        useGpu: Boolean = false
    ): Interpreter {
        val options = Interpreter.Options()
        options.setNumThreads(numberThreads)

        gpuDelegate?.close()
        gpuDelegate = null

        if (useGpu) {
            gpuDelegate = GpuDelegate()
            options.addDelegate(gpuDelegate)
        }

        return Interpreter(loadModelFile(context, modelName), options)
    }

    private fun formatExecutionLog(): String {
        val sb = StringBuilder()
        sb.append("Input Image Size: $imageSize x $imageSize\n")
        sb.append("GPU enabled: $useGPU\n")
        sb.append("Number of threads: $numberThreads\n")
        sb.append("Pre-process execution time: $preprocessTime ms\n")
        sb.append("Model execution time: $imageSegmentationTime ms\n")
        sb.append("Mask flatten time: $maskFlatteningTime ms\n")
        sb.append("Full execution time: $fullTimeExecutionTime ms\n")
        return sb.toString()
    }

    fun destroy() {
        Tasks.call(executorService, Callable {
            try {
                interpreter?.close()
                interpreter = null
                gpuDelegate?.close()
                gpuDelegate = null
            } catch (e: Exception) {
                e.printStackTrace()
            }
        })
    }

    private fun convertBytebufferMaskToBitmap(
        inputBuffer: ByteBuffer,
        imageWidth: Int,
        imageHeight: Int,
        backgroundImage: Bitmap
    ): Triple<Bitmap, Bitmap, Set<Int>> {
        val conf = Bitmap.Config.ARGB_8888
        val maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
        val resultBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
        val itemsFound = HashSet<Int>()
        inputBuffer.rewind()
        val segmentColor = Color.argb(
            (128),
            0,
            255,
            0
        )

        for (y in 0 until imageHeight) {
            for (x in 0 until imageWidth) {
                val index = y * imageWidth + x
                val value = inputBuffer.getFloat(index*4)
                if (value <= 0.0f) {
                    val newPixelColor = ColorUtils.compositeColors(
                        segmentColor,
                        backgroundImage.getPixel(x, y)
                    )
                    resultBitmap.setPixel(x, y, newPixelColor)
                    maskBitmap.setPixel(x, y, segmentColor)
                } else {
                    resultBitmap.setPixel(x, y, backgroundImage.getPixel(x, y))
                }
            }
        }
        return Triple(resultBitmap, maskBitmap, itemsFound)
    }

    companion object {

        private const val TAG = "ImageSegmentationMExec"
        private const val imageSegmentationModel = "segm_model_v5_0065_latency_16fp.tflite"
        private const val imageSize = 256
        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_STD = 255.0f
    }
}