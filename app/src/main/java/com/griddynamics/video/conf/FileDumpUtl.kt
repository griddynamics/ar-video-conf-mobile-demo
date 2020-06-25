package com.griddynamics.video.conf

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStreamWriter
import java.nio.FloatBuffer


class FileDumpUtl {

    fun dumpBytes(context: Context, bytes: ByteArray, fileName: String) =
        try {
            val output = File(context.filesDir, "$fileName.bytes")
            FileOutputStream(output).use {
                it.write(bytes)
            }
        } catch (exc: IOException) {
        }

    fun dumpString(context: Context, bytes: FloatBuffer, fileName: String) {
        val value = StringBuilder()
        for (i in 0 until bytes.limit()) {
            value.append(bytes.get(i).toString()).append("\n")
        }
        try {
            val outputStreamWriter = OutputStreamWriter(
                context.openFileOutput(
                    "$fileName.txt",
                    Context.MODE_PRIVATE
                )
            )
            outputStreamWriter.write(value.toString())
            outputStreamWriter.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}