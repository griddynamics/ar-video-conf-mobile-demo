package com.griddynamics.video.conf

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.drawable.toBitmap
import androidx.core.graphics.scale
import com.griddynamics.video.conf.utils.BuildUtils.Companion.isEmulator
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

        private const val MODEL_IMAGE_SIZE = 64
        private const val MODEL_OUTPUT_BYTES = MODEL_IMAGE_SIZE * MODEL_IMAGE_SIZE * 4
        private const val MODEL_INPUT_BYTES = MODEL_OUTPUT_BYTES * 3

        private const val TAG = "tensorflow_main"
        private const val MODEL_NAME = "segm_model_v11_latency_16fp_better.tflite"
        private const val REQUEST_CODE_PERMISSIONS = 10

        init {
            System.loadLibrary("tensorflow_wrapper")
        }
    }

    private external fun jniInitModel(assetName: String, assetManager: AssetManager): Long
    private external fun jniProcessBitmap(
        input: ByteBuffer,
        result: ByteBuffer,
        inputBitmap: Bitmap,
        backgroundBitmap: Bitmap,
        resultBitmapWidth: Int,
        resultBitmapHeight: Int
    ): Bitmap

    private external fun jniDestroyModel()

    private val coroutineScope = CoroutineScope(Job())
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private lateinit var backgroundImage: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Request camera permissions
        if (allPermissionsGranted()) {
            viewFinder.post {
                startCamera()
            }
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        tvModelInfo.text = MODEL_NAME
        coroutineScope.launch {
            jniInitModel(MODEL_NAME, assets)
        }
    }

    @SuppressLint("NewApi")
    private fun startCamera() {
        val modelInputBuffer = ByteBuffer.allocateDirect(MODEL_INPUT_BYTES).apply {
            order(ByteOrder.nativeOrder())
        }
        val modelOutputBuffer = ByteBuffer.allocateDirect(MODEL_OUTPUT_BYTES).apply {
            order(ByteOrder.nativeOrder())
        }

        backgroundImage = resources.getDrawable(R.drawable.beach_photo, theme).toBitmap()
            .scale(viewFinder.width, viewFinder.height)

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
                            analyze(modelInputBuffer, modelOutputBuffer)
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
                Log.e(TAG, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    var startTimestampMain = System.currentTimeMillis()

    private fun analyze(modelInputBuffer: ByteBuffer, modelOutputBuffer: ByteBuffer) {
        viewFinder.apply {
            bitmap?.let {
                val startTimestamp = System.currentTimeMillis()
                val resultBitmap = jniProcessBitmap(
                    modelInputBuffer,
                    modelOutputBuffer,
                    it,
                    backgroundImage,
                    MODEL_IMAGE_SIZE,
                    MODEL_IMAGE_SIZE
                )
                val diff = System.currentTimeMillis() - startTimestamp

                runOnUiThread {
                    val fps = 1000 / (System.currentTimeMillis() - startTimestampMain)
                    tvFPS.text =
                        "${fps}fps\n(frame update took - $diff ms)"
                    startTimestampMain = System.currentTimeMillis()

                    ivOverlay.setImageBitmap(
                        resultBitmap
                    )
                }
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
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

    override fun onDestroy() {
        jniDestroyModel()
        super.onDestroy()
    }
}