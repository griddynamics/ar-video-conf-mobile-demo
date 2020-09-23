package com.griddynamics.video.conf

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
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
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import com.griddynamics.video.conf.pref.DeviceInfo
import com.griddynamics.video.conf.utils.BuildUtils
import com.griddynamics.video.conf.utils.BuildUtils.Companion.isEmulator
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader
import java.util.*
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
    private lateinit var maskImageSegmentation: ImageSegmentation
    private val imageSegmentationFactory = ImageSegmentationFactory()
    private var timestamp: Long? = null
    private val db = Firebase.firestore


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

        if (DeviceInfo.uuid.isEmpty()) {
            val uuid = UUID.randomUUID()
            DeviceInfo.uuid = uuid.toString()
            val device = hashMapOf(
                "uuid" to DeviceInfo.uuid,
                "info" to BuildUtils.deviceInfo
            )

            db.collection("devices")
                .add(device)
        }
        coroutineScope.launch {
            maskImageSegmentation = imageSegmentationFactory.provideCustom(applicationContext)
            deepLabLite.initialize(applicationContext)
            runOnUiThread {
                btnGD.callOnClick()
            }
        }
    }

    private fun setupButtons() {
        btnNothing.setOnClickListener {
            btnNothing.scaleX = 1.4f
            btnNothing.scaleY = 1.4f
            btnGD.scaleX = 1f
            btnGD.scaleY = 1f

            ivOverlay.visibility = View.INVISIBLE
            ivBack.visibility = View.INVISIBLE
            viewFinder.visibility = View.VISIBLE
            maskImageSegmentation.applyBlur = false
        }
        btnGD.setOnClickListener {
            btnNothing.scaleX = 1f
            btnNothing.scaleY = 1f
            btnGD.scaleX = 1.4f
            btnGD.scaleY = 1.4f

            ivOverlay.visibility = View.VISIBLE
            ivBack.visibility = View.VISIBLE
            viewFinder.visibility = View.INVISIBLE
            maskImageSegmentation.applyBlur = false

            timestamp = null
        }
    }

    @SuppressLint("NewApi")
    private fun startCamera() {
        OpenCVLoader.initDebug()
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
        viewFinder.bitmap?.let { bitmap ->
            val result = maskImageSegmentation.execute(
                bitmap
            )
            runOnUiThread {
                ivOverlay.setImageBitmap(
                    result
                )
                timestamp?.let {
                    val log = hashMapOf(
                        "uuid" to DeviceInfo.uuid,
                        "model" to DeviceInfo.model,
                        "tag" to "fps",
                        "info" to System.currentTimeMillis() - it
                    )
                    db.collection("logs").add(log)
                }
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