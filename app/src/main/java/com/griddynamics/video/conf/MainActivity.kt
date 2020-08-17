package com.griddynamics.video.conf

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.*
import androidx.core.graphics.drawable.toBitmap
import com.griddynamics.video.conf.utils.BuildUtils.Companion.isEmulator
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.launch
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
    private var backgroundImage: Bitmap? = null
    private val i256 = 256
    private lateinit var imageSegmentation: ImageSegmentation
    private val imageSegmentationFactory = ImageSegmentationFactory()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        if (allPermissionsGranted()) {
            viewFinder.post {
                startCamera()
            }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        setupButtons()
        coroutineScope.launch {
            imageSegmentation = imageSegmentationFactory.provideCustom(applicationContext)
        }
    }

    private fun setupButtons() {
        btnNothing.setOnClickListener {
            btnNothing.scaleX = 1.4f
            btnNothing.scaleY = 1.4f
            btnBeach.scaleX = 1f
            btnBeach.scaleY = 1f
            btnGD.scaleX = 1f
            btnGD.scaleY = 1f

            backgroundImage = null
        }
        btnBeach.setOnClickListener {
            btnNothing.scaleX = 1f
            btnNothing.scaleY = 1f
            btnBeach.scaleX = 1.4f
            btnBeach.scaleY = 1.4f
            btnGD.scaleX = 1f
            btnGD.scaleY = 1f

            backgroundImage = getDrawable(R.drawable.beach)?.toBitmap()!!.scale(256, 256)
                .scale(256, 256)
        }
        btnGD.setOnClickListener {
            btnNothing.scaleX = 1f
            btnNothing.scaleY = 1f
            btnBeach.scaleX = 1f
            btnBeach.scaleY = 1f
            btnGD.scaleX = 1.4f
            btnGD.scaleY = 1.4f

            backgroundImage =
                getDrawable(R.drawable.gd)?.toBitmap()!!.scale(256, 256)
        }

        btnGD.callOnClick()
    }

    @SuppressLint("NewApi")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            preview = Preview.Builder()
                .build()
            imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(
                        Executors.newWorkStealingPool(),
                        ImageAnalysis.Analyzer { image ->
                            analyze()
                            image.close()
                        })
                }
            // Select back camera
            val cameraSelector =
                CameraSelector.Builder()
                    .requireLensFacing(if (isEmulator()) CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT)
                    .build()

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                preview?.setSurfaceProvider(viewFinder.createSurfaceProvider())
            } catch (exc: Exception) {
                Log.e(javaClass.name, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze() {
        backgroundImage?.let {
            viewFinder.bitmap?.let { bitmap ->
                val startTimestamp = System.currentTimeMillis()

                val result = imageSegmentation.execute(
                    bitmap, bitmap.width,
                    bitmap.height
                )
                val diff = System.currentTimeMillis() - startTimestamp
                val scaledBitmap = result.scale(i256, i256)
                for (y in 0 until i256) {
                    for (x in 0 until i256) {
                        val segmentColor = Color.argb(
                            scaledBitmap[x, y].alpha,
                            it[x, y].red,
                            it[x, y].green,
                            it[x, y].blue
                        )

                        scaledBitmap.setPixel(x, y, segmentColor)
                    }
                }

                runOnUiThread {
                    ivOverlay.setImageBitmap(
                        scaledBitmap
                    )
                }
            }
        }
        runOnUiThread {
            ivOverlay.visibility =
                if (backgroundImage == null) View.GONE else View.VISIBLE

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