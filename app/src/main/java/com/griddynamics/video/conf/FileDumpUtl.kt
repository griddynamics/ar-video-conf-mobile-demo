package com.griddynamics.video.conf

import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class FileDumpUtl {

    fun dumpBytes(context: Context, bytes: ByteArray, fileName: String) =
        try {
            val output = File(context.filesDir, "$fileName.bytes")
            FileOutputStream(output).use {
                it.write(bytes)
            }
        } catch (exc: IOException) {
        }
}