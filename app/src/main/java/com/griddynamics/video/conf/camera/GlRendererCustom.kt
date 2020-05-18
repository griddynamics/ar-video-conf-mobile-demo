package com.griddynamics.video.conf.camera

import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.effect.Effect
import android.media.effect.EffectContext
import android.media.effect.EffectFactory
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.opengl.GLUtils
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.core.graphics.scale
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor1
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutorCustom
import com.griddynamics.video.conf.toBitmap
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class GlRendererCustom(
    private val gLSurfaceView: GLSurfaceView,
    private val imageSegmentation: ImageSegmentationModelExecutorCustom,
    private val width: Int,
    private val height: Int
) : GLSurfaceView.Renderer,
    ImageAnalysis.Analyzer {
    private val textures = IntArray(2)
    private var square: Square? = null


    private var effectContext: EffectContext? = null
    private var effect: Effect? = null

    private var image: Bitmap? = null

    @Synchronized
    fun setImage(image: Bitmap) {
        //this.image?.recycle()

        this.image = image
    }

    override fun onDrawFrame(p0: GL10?) {
        generateSquare()
        generateTexture()

        if (effectContext == null) {
            effectContext = EffectContext.createWithCurrentGlContext()
        }

        effect?.release()
        applyEffect()

        square?.draw(textures[1])
    }

    override fun onSurfaceChanged(p0: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        generateSquare()
    }

    override fun onSurfaceCreated(p0: GL10?, p1: EGLConfig?) {
    }

    private fun generateSquare() {
        if (square == null) {
            square = Square()
        }
    }

    private fun generateTexture() {
        GLES20.glGenTextures(2, textures, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textures[0])


        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_S,
            GLES20.GL_CLAMP_TO_EDGE
        )
        GLES20.glTexParameteri(
            GLES20.GL_TEXTURE_2D,
            GLES20.GL_TEXTURE_WRAP_T,
            GLES20.GL_CLAMP_TO_EDGE
        )

        image?.run {
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, this, 0)
            // GLES20.glTexImage2D(GL10.GL_TEXTURE_2D, 0, GL10.GL_RGBA, width, height, 0, GL10.GL_RGBA, GL10.GL_UNSIGNED_BYTE, ByteBuffer.wrap(RgbBytes))
        }
    }

    private fun applyEffect() {
        val image = this.image
        val effectContext = this.effectContext
        if (image != null && effectContext != null) {
            val factory = effectContext.factory
            effect = factory.createEffect(EffectFactory.EFFECT_AUTOFIX).apply {
                apply(textures[0], image.width, image.height, textures[1])
            }
        }
    }

    var counter = 0
    override fun analyze(image: ImageProxy, rotationDegrees: Int) {
        counter++
        if (counter%2 == 0) {
            return
        }
        val matrix = Matrix()
        matrix.postRotate(rotationDegrees.toFloat())

        val b = image.image!!.toBitmap()
        val bm = Bitmap.createBitmap(b, 0, 0, b.width, b.height, matrix, true)
        //setImage(bm)
        imageSegmentation?.executeAsync(bm)
            ?.addOnSuccessListener { result ->
                result?.bitmapResult ?: return@addOnSuccessListener
                setImage(result?.bitmapResult.scale(width, height, false))
                gLSurfaceView.requestRender()
            }
            ?.addOnFailureListener {

            }
    }
}
