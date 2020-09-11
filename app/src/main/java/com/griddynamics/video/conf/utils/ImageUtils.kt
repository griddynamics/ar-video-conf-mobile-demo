package com.griddynamics.video.conf.utils

import android.graphics.*
import android.media.ExifInterface
import android.media.Image
import android.os.Build
import androidx.annotation.RequiresApi
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

abstract class ImageUtils {
    companion object {

        /**
         * Helper function used to convert an EXIF orientation enum into a transformation matrix
         * that can be applied to a bitmap.
         *
         * @param orientation - One of the constants from [ExifInterface]
         */
        private fun decodeExifOrientation(orientation: Int): Matrix {
            val matrix = Matrix()

            // Apply transformation corresponding to declared EXIF orientation
            when (orientation) {
                ExifInterface.ORIENTATION_NORMAL, ExifInterface.ORIENTATION_UNDEFINED -> Unit
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90F)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180F)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270F)
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1F, 1F)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1F, -1F)
                ExifInterface.ORIENTATION_TRANSPOSE -> {
                    matrix.postScale(-1F, 1F)
                    matrix.postRotate(270F)
                }
                ExifInterface.ORIENTATION_TRANSVERSE -> {
                    matrix.postScale(-1F, 1F)
                    matrix.postRotate(90F)
                }

                // Error out if the EXIF orientation is invalid
                else -> throw IllegalArgumentException("Invalid orientation: $orientation")
            }

            // Return the resulting matrix
            return matrix
        }

        /**
         * sets the Exif orientation of an image.
         * this method is used to fix the exit of pictures taken by the camera
         *
         * @param filePath - The image file to change
         * @param value - the orientation of the file
         */
        fun setExifOrientation(
            filePath: String,
            value: String
        ) {
            val exif = ExifInterface(filePath)
            exif.setAttribute(
                ExifInterface.TAG_ORIENTATION, value
            )
            exif.saveAttributes()
        }

        /** Transforms rotation and mirroring information into one of the [ExifInterface] constants */
        fun computeExifOrientation(rotationDegrees: Int, mirrored: Boolean) = when {
            rotationDegrees == 0 && !mirrored -> ExifInterface.ORIENTATION_NORMAL
            rotationDegrees == 0 && mirrored -> ExifInterface.ORIENTATION_FLIP_HORIZONTAL
            rotationDegrees == 180 && !mirrored -> ExifInterface.ORIENTATION_ROTATE_180
            rotationDegrees == 180 && mirrored -> ExifInterface.ORIENTATION_FLIP_VERTICAL
            rotationDegrees == 270 && mirrored -> ExifInterface.ORIENTATION_TRANSVERSE
            rotationDegrees == 90 && !mirrored -> ExifInterface.ORIENTATION_ROTATE_90
            rotationDegrees == 90 && mirrored -> ExifInterface.ORIENTATION_TRANSPOSE
            rotationDegrees == 270 && mirrored -> ExifInterface.ORIENTATION_ROTATE_270
            rotationDegrees == 270 && !mirrored -> ExifInterface.ORIENTATION_TRANSVERSE
            else -> ExifInterface.ORIENTATION_UNDEFINED
        }

        /**
         * Decode a bitmap from a file and apply the transformations described in its EXIF data
         *
         * @param file - The image file to be read using [BitmapFactory.decodeFile]
         */
        fun decodeBitmap(file: File): Bitmap {
            // First, decode EXIF data and retrieve transformation matrix
            val exif = ExifInterface(file.absolutePath)
            val transformation =
                decodeExifOrientation(
                    exif.getAttributeInt(
                        ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_ROTATE_90
                    )
                )

            // Read bitmap using factory methods, and transform it using EXIF data
            val options = BitmapFactory.Options()
            val bitmap = BitmapFactory.decodeFile(file.absolutePath, options)
            return Bitmap.createBitmap(
                BitmapFactory.decodeFile(file.absolutePath),
                0, 0, bitmap.width, bitmap.height, transformation, true
            )
        }

        fun scaleBitmapAndKeepRatio(
            targetBmp: Bitmap,
            reqHeightInPixels: Int,
            reqWidthInPixels: Int
        ): Bitmap {
            if (targetBmp.height == reqHeightInPixels && targetBmp.width == reqWidthInPixels) {
                return targetBmp
            }
            return Bitmap.createScaledBitmap(targetBmp, reqWidthInPixels, reqHeightInPixels, true)
        }

        @RequiresApi(Build.VERSION_CODES.Q)
        fun bitmapToByteBuffer(
            bitmapIn: Bitmap,
            width: Int,
            height: Int,
            mean: Float = 0.0f,
            std: Float = 255.0f
        ): ByteBuffer {
            val bitmap = scaleBitmapAndKeepRatio(bitmapIn, width, height)
            val inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4)
            inputImage.order(ByteOrder.nativeOrder())
            inputImage.rewind()

            for (y in 0 until height) {
                for (x in 0 until width) {
                    val value = bitmap.getPixel(x, y)
                    inputImage.putFloat(((value shr 16 and 0xFF) - mean) / std)
                    inputImage.putFloat(((value shr 8 and 0xFF) - mean) / std)
                    inputImage.putFloat(((value and 0xFF) - mean) / std)
                }
            }

            inputImage.rewind()
            return inputImage
        }

        fun bitmapToNormalizedArray(
            bitmapIn: Bitmap,
            width: Int,
            height: Int,
            mean: Float = 0.0f,
            std: Float = 255.0f
        ): Array<Array<Array<FloatArray>>> {
            val bitmap = scaleBitmapAndKeepRatio(bitmapIn, width, height)
            val arr = Array(1) { Array(width) { Array(width) { FloatArray(3) } } }
            val inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4)
            inputImage.order(ByteOrder.nativeOrder())
            inputImage.rewind()

            val intValues = IntArray(width * height)
            bitmap.getPixels(intValues, 0, width, 0, 0, width, height)
            var pixel = 0
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val value = intValues[pixel++]

                    // Normalize channel values to [-1.0, 1.0]. This requirement varies by
                    // model. For example, some models might require values to be normalized
                    // to the range [0.0, 1.0] instead.
/*                    if((((value shr 16 and 0xFF) - mean) / std) < 0 || (((value shr 8 and 0xFF) - mean) / std) < 0 || (((value and 0xFF) - mean) / std) < 0) {
                        throw Exception("Normalize")
                    }*/
                    arr[0][y][x][0] = ((value shr 16 and 0xFF) - mean) / std
                    arr[0][y][x][1] = ((value shr 8 and 0xFF) - mean) / std
                    arr[0][y][x][2] = ((value and 0xFF) - mean) / std
                }
            }

            inputImage.rewind()

            return arr
        }

        fun bitmapToByteBuffer1(
            bitmapIn: Bitmap,
            width: Int,
            height: Int,
            mean: Float = 0.0f,
            std: Float = 255.0f
        ): ByteBuffer {
            val bitmap = scaleBitmapAndKeepRatio(bitmapIn, width, height)
            val inputImage = ByteBuffer.allocateDirect(1 * width * height * 3 * 4)
            inputImage.order(ByteOrder.nativeOrder())
            inputImage.rewind()

            val intValues = IntArray(width * height)
            bitmap.getPixels(intValues, 0, width, 0, 0, width, height)
            var pixel = 0
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val value = intValues[pixel++]

                    // Normalize channel values to [-1.0, 1.0]. This requirement varies by
                    // model. For example, some models might require values to be normalized
                    // to the range [0.0, 1.0] instead.
                    inputImage.putFloat(((value shr 16 and 0xFF) - mean) / std)
                    inputImage.putFloat(((value shr 8 and 0xFF) - mean) / std)
                    inputImage.putFloat(((value and 0xFF) - mean) / std)
                }
            }

            inputImage.rewind()
            return inputImage
        }

        fun createEmptyBitmap(imageWidth: Int, imageHeigth: Int, color: Int = 0): Bitmap {
            val ret = Bitmap.createBitmap(imageWidth, imageHeigth, Bitmap.Config.RGB_565)
            if (color != 0) {
                ret.eraseColor(color)
            }
            return ret
        }
    }
}

fun Image.toJPEGBitmap(rotation: Int): Bitmap {
    val buffer = planes[0].buffer
    buffer.rewind()
    val bytes = ByteArray(buffer.capacity())
    buffer.get(bytes)
    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, null)
    val matrix = Matrix()
    matrix.postRotate(rotation.toFloat())
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

fun ImageProxy.toYUV420Bitmap(): Bitmap {
    val i256 = 256
    val yBuffer = planes[0].buffer // Y
    val uBuffer = planes[1].buffer // U
    val vBuffer = planes[2].buffer // V

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, this.width, this.height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, i256, i256), 100, out)
    val imageBytes = out.toByteArray()
    val bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    val matrix = Matrix()
    matrix.postRotate(90f)
    return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
}

/**
 * Returns a transformation matrix from one reference frame into another. Handles cropping (if
 * maintaining aspect ratio is desired) and rotation.
 *
 * @param srcWidth Width of source frame.
 * @param srcHeight Height of source frame.
 * @param dstWidth Width of destination frame.
 * @param dstHeight Height of destination frame.
 * @param applyRotation Amount of rotation to apply from one frame to another. Must be a multiple
 * of 90.
 * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant,
 * cropping the image if necessary.
 * @return The transformation fulfilling the desired requirements.
 */
fun getTransformationMatrix(
    srcWidth: Int,
    srcHeight: Int,
    dstWidth: Int,
    dstHeight: Int,
    applyRotation: Int,
    maintainAspectRatio: Boolean
): Matrix? {
    val matrix = Matrix()
    if (applyRotation != 0) {
        if (applyRotation % 90 != 0) {
        }

        // Translate so center of image is at origin.
        matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)

        // Rotate around origin.
        matrix.postRotate(applyRotation.toFloat())
    }

    // Account for the already applied rotation, if any, and then determine how
    // much scaling is needed for each axis.
    val transpose = (Math.abs(applyRotation) + 90) % 180 == 0
    val inWidth = if (transpose) srcHeight else srcWidth
    val inHeight = if (transpose) srcWidth else srcHeight

    // Apply scaling if necessary.
    if (inWidth != dstWidth || inHeight != dstHeight) {
        val scaleFactorX = dstWidth / inWidth.toFloat()
        val scaleFactorY = dstHeight / inHeight.toFloat()
        if (maintainAspectRatio) {
            // Scale by minimum factor so that dst is filled completely while
            // maintaining the aspect ratio. Some image may fall off the edge.
            val scaleFactor = Math.max(scaleFactorX, scaleFactorY)
            matrix.postScale(scaleFactor, scaleFactor)
        } else {
            // Scale exactly to fill dst from src.
            matrix.postScale(scaleFactorX, scaleFactorY)
        }
    }
    if (applyRotation != 0) {
        // Translate back from origin centered reference to destination frame.
        matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
    }
    return matrix
}

/**
 * Returns a transformation matrix from one reference frame into another. Handles cropping (if
 * maintaining aspect ratio is desired) and rotation.
 *
 * @param srcWidth Width of source frame.
 * @param srcHeight Height of source frame.
 * @param dstWidth Width of destination frame.
 * @param dstHeight Height of destination frame.
 * @param applyRotation Amount of rotation to apply from one frame to another. Must be a multiple
 * of 90.
 * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant,
 * cropping the image if necessary.
 * @return The transformation fulfilling the desired requirements.
 */
fun getTransformationMatrixContain(
    srcWidth: Int,
    srcHeight: Int,
    dstWidth: Int,
    dstHeight: Int,
    applyRotation: Int,
    maintainAspectRatio: Boolean
): Matrix {
    val matrix = Matrix()
    if (applyRotation != 0) {
        if (applyRotation % 90 != 0) {
        }

        // Translate so center of image is at origin.
        matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)

        // Rotate around origin.
        matrix.postRotate(applyRotation.toFloat())
    }

    // Account for the already applied rotation, if any, and then determine how
    // much scaling is needed for each axis.
    val transpose = (Math.abs(applyRotation) + 90) % 180 == 0
    val inWidth = if (transpose) srcHeight else srcWidth
    val inHeight = if (transpose) srcWidth else srcHeight

    // Apply scaling if necessary.
    if (inWidth != dstWidth || inHeight != dstHeight) {
        val scaleFactorX = dstWidth / inWidth.toFloat()
        val scaleFactorY = dstHeight / inHeight.toFloat()
        if (maintainAspectRatio) {
            // Scale by minimum factor so that dst is filled completely while
            // maintaining the aspect ratio. Some image may fall off the edge.
            val scaleFactor = Math.min(scaleFactorX, scaleFactorY)
            matrix.postScale(scaleFactor, scaleFactor)
            matrix.postTranslate(
                (dstWidth - scaleFactor * inWidth) / 32f,
                (dstHeight - scaleFactor * inHeight) / 2f
            )

//        if (scaleFactorY > scaleFactorX) {
//          matrix.postTranslate((dstWidth - (scaleFactor * inWidth)) / 2f, (dstHeight - (scaleFactor * inHeight)) / 2f);
//        } else {
//          matrix.postTranslate(, 0);
//        }
        } else {
            // Scale exactly to fill dst from src.
            matrix.postScale(scaleFactorX, scaleFactorY)
        }
    }
    if (applyRotation != 0) {
        // Translate back from origin centered reference to destination frame.
        matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
    }

//    matrix.postTranslate(0, dstHeight / 2.0f);
    return matrix
}
