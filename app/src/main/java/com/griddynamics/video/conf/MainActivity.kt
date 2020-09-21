package com.griddynamics.video.conf

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.DisplayMetrics
import android.util.Log
import android.util.Rational
import android.util.Size
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import com.griddynamics.video.conf.palm.PalmDetectionActivity
import com.griddynamics.video.conf.utils.toYUV420Bitmap
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        private const val REQUEST_CODE_PERMISSIONS = 10

    }

    private val coroutineScope = CoroutineScope(Job())
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private val i256 = 256
    private val deepLabLite: DeepLabLite = DeepLabLite()
    private lateinit var imageSegmentation: ImageSegmentation
    private val imageSegmentationFactory = ImageSegmentationFactory()
    private var useDeeplab = false
    private var timestamp = System.currentTimeMillis()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (allPermissionsGranted()) {
            viewFinder.post {
                startCamera()
            }
            viewFinder.addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
                updateTransform()
            }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        setupButtons()
        coroutineScope.launch {
            imageSegmentation = imageSegmentationFactory.provideCustom(applicationContext)
            deepLabLite.initialize(applicationContext)
            runOnUiThread {
                btnNothing.callOnClick()
            }
        }
    }

    private fun setupButtons() {
        btnNothing.setOnClickListener {
            btnNothing.scaleX = 1.4f
            btnNothing.scaleY = 1.4f
            btnBlur.scaleX = 1f
            btnBlur.scaleY = 1f
            btnGD.scaleX = 1f
            btnGD.scaleY = 1f
            btnDL.scaleX = 1f
            btnDL.scaleY = 1f

            ivOverlay.visibility = View.INVISIBLE
            ivBack.visibility = View.INVISIBLE
            viewFinder.visibility = View.VISIBLE
            imageSegmentation.applyBlur = false
            useDeeplab = false
        }
        btnGD.setOnClickListener {
            btnNothing.scaleX = 1f
            btnNothing.scaleY = 1f
            btnBlur.scaleX = 1f
            btnBlur.scaleY = 1f
            btnGD.scaleX = 1.4f
            btnGD.scaleY = 1.4f
            btnDL.scaleX = 1f
            btnDL.scaleY = 1f

            ivOverlay.visibility = View.VISIBLE
            ivBack.visibility = View.VISIBLE
            //    viewFinder.visibility = View.INVISIBLE
            imageSegmentation.applyBlur = false
            useDeeplab = false
        }
        btnBlur.setOnClickListener {
            btnNothing.scaleX = 1f
            btnNothing.scaleY = 1f
            btnBlur.scaleX = 1.4f
            btnBlur.scaleY = 1.4f
            btnGD.scaleX = 1f
            btnGD.scaleY = 1f
            btnDL.scaleX = 1f
            btnDL.scaleY = 1f

            ivOverlay.visibility = View.VISIBLE
            ivBack.visibility = View.INVISIBLE
            viewFinder.visibility = View.INVISIBLE
            imageSegmentation.applyBlur = true
            useDeeplab = false
        }
        btnDL.setOnClickListener {
            startActivity(Intent(this, PalmDetectionActivity::class.java))
        }
    }

    private fun startCamera() {
        val metrics = DisplayMetrics().also { viewFinder.display.getRealMetrics(it) }
        val screenSize = Size(metrics.widthPixels, metrics.heightPixels)
        val screenAspectRatio = Rational(metrics.widthPixels, metrics.heightPixels)
        preview = Preview(
            PreviewConfig.Builder().apply {
                setTargetResolution(screenSize)
                setLensFacing(CameraX.LensFacing.FRONT)
            }.build()
        )
        preview?.setOnPreviewOutputUpdateListener {
            // To update the SurfaceTexture, we have to remove it and r
            val parent = viewFinder.parent as ViewGroup
            parent.removeView(viewFinder)
            parent.addView(viewFinder, 0)
            viewFinder.surfaceTexture = it.surfaceTexture
           // updateTransform()
        }
        val imageCaptureConfig = ImageCaptureConfig.Builder().apply {
            setTargetResolution(screenSize)
            setLensFacing(CameraX.LensFacing.FRONT)
            setCaptureMode(ImageCapture.CaptureMode.MIN_LATENCY)
        }.build()

        val imageCapture = ImageCapture(imageCaptureConfig)

        val imageAnalysisConfig = ImageAnalysisConfig.Builder().apply {
           // setTargetResolution(Size(256, 256))
            setTargetResolution(screenSize)
            setLensFacing(CameraX.LensFacing.FRONT)
            setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
        }.build()
        val imageAnalysis = ImageAnalysis(imageAnalysisConfig)
        imageAnalysis.setAnalyzer(
            Executors.newWorkStealingPool(),
            ImageAnalysis.Analyzer { image, rotationDegrees ->
                analyze(viewFinder.bitmap)
            })
        CameraX.bindToLifecycle(this, imageAnalysis, imageCapture, preview)
    }

    private fun updateTransform() {
        val matrix = Matrix()

        // Compute the center of the view finder
        val centerX = viewFinder.width / 2f
        val centerY = viewFinder.height / 2f

        // Correct preview output to account for display rotation
        val rotationDegrees = when (viewFinder.display.rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> return
        }
        matrix.postRotate(-rotationDegrees.toFloat(), centerX, centerY)

        // Finally, apply transformations to our TextureView
        viewFinder.setTransform(matrix)
    }

    private fun analyze(bitmap: Bitmap?) {
        bitmap?.let { _ ->
            val result = if (useDeeplab) {
                deepLabLite.segment(bitmap.scale(257, 257))

            } else {
                imageSegmentation.execute(
                    bitmap, bitmap.width,
                    bitmap.height
                )
            }
            runOnUiThread {
                ivOverlay.setImageBitmap(
                    result
                )
                timestamp = System.currentTimeMillis()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onDestroy() {
        imageSegmentationFactory.destroy()
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray
    ) {
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(
                    this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }
}