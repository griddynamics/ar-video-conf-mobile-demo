package com.griddynamics.video.conf

import android.content.Context
import com.griddynamics.video.conf.pref.DeviceInfo
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ImageSegmentationFactory(private val logger: Logger) {

    enum class MaskModel(val filename: String) {

        mask32Model("semsegm_of8000_latency_16fp.tflite") {
            override fun imageSize() = 32
        },
        mask64Model("fil_8_shape_64x64_latency_16fp.tflite") {
            override fun imageSize() = 64
        },
        mask128_8_Model("fil_8_shape_128x128_latency_16fp.tflite") {
            override fun imageSize() = 128
        },
        mask128_12_Model("fil_12_shape_128x128_latency_16fp.tflite") {
            override fun imageSize() = 128
        },
        mask128_16_Model("fil_16_shape_128x128_latency_16fp.tflite") {
            override fun imageSize() = 128
        };

        companion object {
            fun getByFilename(name: String) =
                values().find { it.filename == name } ?: mask32Model
        }

        abstract fun imageSize(): Int
    }

    private lateinit var interpreter: Interpreter

    fun provideCustom(context: Context, filename: String): ImageSegmentation {
        val model = MaskModel.getByFilename(filename)
        interpreter = getInterpreter(context, model.filename)
        DeviceInfo.model = model.filename
        return ImageSegmentation(interpreter, logger, model.imageSize())
    }

    fun destroy() {
        interpreter.close()
    }

    @Throws(IOException::class)
    private fun getInterpreter(
        context: Context,
        modelName: String
    ): Interpreter {
        val options = Interpreter.Options()
        return Interpreter(loadModelFile(context, modelName), options)
    }

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
}