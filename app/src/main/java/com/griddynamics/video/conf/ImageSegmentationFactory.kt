package com.griddynamics.video.conf

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class ImageSegmentationFactory {

    companion object {
        const val maskModel = "segm_model_32_32_8_0.07_latency_16fp.tflite"
    }

    private lateinit var interpreter: Interpreter

    fun provideCustom(context: Context): ImageSegmentation {
        interpreter = getInterpreter(context, maskModel)
        return ImageSegmentation(interpreter)
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