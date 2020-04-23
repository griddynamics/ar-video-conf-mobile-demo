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
import kotlin.random.Random

/**
 * Class responsible to run the Image Segmentation model.
 * more information about the DeepLab model being used can
 * be found here:
 * https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html
 * https://www.tensorflow.org/lite/models/segmentation/overview
 * https://github.com/tensorflow/models/tree/master/research/deeplab
 *
 * Label names: 'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
 * 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
 * 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
 */
class ImageSegmentationModelExecutor(
    private val context: Context,
    private var useGPU: Boolean = false
) {
    private var gpuDelegate: GpuDelegate? = null

    private var segmentationMasks: ByteBuffer? = null
    private var interpreter: Interpreter? = null

    private var fullTimeExecutionTime = 0L
    private var preprocessTime = 0L
    private var imageSegmentationTime = 0L
    private var maskFlatteningTime = 0L

    private var numberThreads = 2

    private val executorService: ExecutorService = Executors.newSingleThreadExecutor()

    fun initialize(): Task<Nothing?> {
        return Tasks.call(executorService, Callable {
            interpreter = getInterpreter(context, imageSegmentationModel, useGPU)
            segmentationMasks =
                ByteBuffer.allocateDirect(1 * imageSize * imageSize * NUM_CLASSES * 4)
            segmentationMasks!!.order(ByteOrder.nativeOrder())
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
        interpreter!!.run(contentArray, segmentationMasks)
        imageSegmentationTime = SystemClock.uptimeMillis() - imageSegmentationTime
        Log.d(TAG, "Time to run the model $imageSegmentationTime")

        maskFlatteningTime = SystemClock.uptimeMillis()
        val (maskImageApplied, maskOnly, itemsFound) =
            convertBytebufferMaskToBitmap(
                segmentationMasks!!,
                imageSize,
                imageSize, scaledBitmap,
                segmentColors
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
        backgroundImage: Bitmap,
        colors: IntArray
    ): Triple<Bitmap, Bitmap, Set<Int>> {
        val conf = Bitmap.Config.ARGB_8888
        val maskBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
        val resultBitmap = Bitmap.createBitmap(imageWidth, imageHeight, conf)
        val scaledBackgroundImage =
            ImageUtils.scaleBitmapAndKeepRatio(
                backgroundImage,
                imageWidth,
                imageHeight
            )
        val mSegmentBits = Array(imageWidth) { IntArray(imageHeight) }
        val itemsFound = HashSet<Int>()
        inputBuffer.rewind()

        for (y in 0 until imageHeight) {
            for (x in 0 until imageWidth) {
                var maxVal = 0f
                mSegmentBits[x][y] = 0

                for (c in 0 until NUM_CLASSES) {
                    val value = inputBuffer
                        .getFloat((y * imageWidth * NUM_CLASSES + x * NUM_CLASSES + c) * 4)
                    if (c == 0 || value > maxVal) {
                        maxVal = value
                        mSegmentBits[x][y] = c
                    }
                }

                itemsFound.add(mSegmentBits[x][y])
                val newPixelColor = ColorUtils.compositeColors(
                    colors[mSegmentBits[x][y]],
                    scaledBackgroundImage.getPixel(x, y)
                )
                resultBitmap.setPixel(x, y, newPixelColor)
                maskBitmap.setPixel(x, y, colors[mSegmentBits[x][y]])
            }
        }

        return Triple(resultBitmap, maskBitmap, itemsFound)
    }

    companion object {

        private const val TAG = "ImageSegmentationMExec"
        private const val imageSegmentationModel = "deeplabv3_257_mv_gpu.tflite"
        private const val imageSize = 257
        const val NUM_CLASSES = 21
        private const val IMAGE_MEAN = 128.0f
        private const val IMAGE_STD = 128.0f

        val segmentColors = IntArray(NUM_CLASSES)
        val labelsArrays = arrayOf(
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
            "person", "potted plant", "sheep", "sofa", "train", "tv"
        )

        init {
            val random = Random(System.currentTimeMillis())
            segmentColors[0] = Color.TRANSPARENT
            for (i in 1 until NUM_CLASSES) {
                segmentColors[i] = Color.argb(
                    (128),
                    getRandomRGBInt(
                        random
                    ),
                    getRandomRGBInt(
                        random
                    ),
                    getRandomRGBInt(
                        random
                    )
                )
            }
        }

        private fun getRandomRGBInt(random: Random) = (255 * random.nextFloat()).toInt()
    }
}