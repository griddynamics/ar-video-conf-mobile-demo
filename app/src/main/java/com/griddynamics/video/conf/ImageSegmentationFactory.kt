package com.griddynamics.video.conf

import android.content.Context
import com.griddynamics.video.conf.pref.DeviceInfo
import com.griddynamics.video.conf.pref.Settings
import org.tensorflow.lite.Interpreter
import java.io.*
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Paths

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

    fun provideCustom(context: Context): ImageSegmentation {
        interpreter = getInterpreter(context, Settings.modelName)
        return ImageSegmentation(interpreter, logger, Settings.modelSize)
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

  //      if (modelFile.contains(SettingsBottomDialogFragment.modelPrefix)) {
  //          val path = Paths.get(context.filesDir.absolutePath, modelFile)
  //          val file = RandomAccessFile(File(path.toUri()), "rw")
  //          val ch = file.channel
  //          val size = ch.size()
  //          val retFile = ch.map(FileChannel.MapMode.READ_ONLY, 0, size)
  //          ch.close()
  //          return retFile
  //      } else {
  //          val model = MaskModel.getByFilename(modelFile)
  //          model?.let {
  //              val fileDescriptor = context.assets.openFd(it.filename)
  //              val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
  //              val fileChannel = inputStream.channel
  //              val startOffset = fileDescriptor.startOffset
  //              val declaredLength = fileDescriptor.declaredLength
  //              val retFile =
  //                  fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
  //              fileDescriptor.close()
  //              return retFile
  //          }
  //          throw NotImplementedError()
  //      }
  //
        try {
            val fileDescriptor = context.assets.openFd(modelFile)
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val retFile = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
            fileDescriptor.close()
            return retFile
        }
        catch (ex: FileNotFoundException){
            val path = Paths.get(context.filesDir.absolutePath, modelFile)
            val file = RandomAccessFile(File(path.toUri()), "rw")
            val ch = file.channel
            val size = ch.size()
            val retFile = ch.map(FileChannel.MapMode.READ_ONLY, 0, size)
            ch.close()
            return retFile
        }
    }
}