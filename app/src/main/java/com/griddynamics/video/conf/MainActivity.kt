package com.griddynamics.video.conf

import android.Manifest
import android.annotation.SuppressLint
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
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
import com.google.android.material.snackbar.Snackbar
import com.google.firebase.firestore.ktx.firestore
import com.google.firebase.ktx.Firebase
import com.google.firebase.storage.ktx.storage
import com.griddynamics.video.conf.pref.DeviceInfo
import com.griddynamics.video.conf.pref.Settings
import com.griddynamics.video.conf.utils.BuildUtils
import com.griddynamics.video.conf.utils.BuildUtils.Companion.isEmulator
import com.griddynamics.video.conf.utils.safeClick
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.Runnable
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
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
    private var maskImageSegmentation: ImageSegmentation? = null
    private val logger = Logger(Firebase.firestore)
    private val imageSegmentationFactory = ImageSegmentationFactory(logger)
    private var timestamp: Long? = null
    private var sendBitmapToStorage = false

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

            logger.log(device, "devices")
        }
        setInterpreter()
    }

    private fun setInterpreter() {
        coroutineScope.launch {
            maskImageSegmentation =
                imageSegmentationFactory.provideCustom(applicationContext, Settings.modelName)
        }
    }

    private fun setupButtons() {
        navigation.selectedItemId = R.id.action_mask
        navigation.setOnNavigationItemSelectedListener {
            when (it.itemId) {
                R.id.action_clear -> {
                    ivOverlay.visibility = View.INVISIBLE
                    ivBack.visibility = View.INVISIBLE
                    viewFinder.visibility = View.VISIBLE
                    fab.visibility = View.GONE
                }
                else -> {
                    ivOverlay.visibility = View.VISIBLE
                    ivBack.visibility = View.VISIBLE
                    viewFinder.visibility = View.INVISIBLE
                    fab.visibility = View.VISIBLE
                    timestamp = null
                }
            }
            true
        }
        btnSettings.safeClick({
            SettingsBottomDialogFragment(logger, object :
                SettingsBottomDialogFragment.OnModelChange {
                override fun onModelChange() {
                    cameraProvider.unbindAll()

                    Handler().postDelayed({
                        imageSegmentationFactory.destroy()
                        setInterpreter()
                        Handler().postDelayed({
                            startCamera()
                        }, 1000)
                    }, 1000)
                }
            }).show(supportFragmentManager, null)
        })
        fab.safeClick({
            fab.isEnabled = false
            sendBitmapToStorage = true
        })
    }

    lateinit var cameraProvider: ProcessCameraProvider

    @SuppressLint("NewApi")
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener(Runnable {
            cameraProvider = cameraProviderFuture.get()

            preview = Preview.Builder()
                .build()
            imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(
                        Executors.newWorkStealingPool(),
                        { image ->
                            analyze()
                            image.close()
                        })
                }
            val cameraSelector =
                CameraSelector.Builder()
                    .requireLensFacing(if (isEmulator()) CameraSelector.LENS_FACING_BACK else CameraSelector.LENS_FACING_FRONT)
                    .build()

            try {
                cameraProvider.unbindAll()
                camera = cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageAnalyzer
                )
                preview?.setSurfaceProvider(viewFinder.surfaceProvider)
            } catch (exc: Exception) {
                Log.e(javaClass.name, "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyze() {
        viewFinder.bitmap?.let { bitmap ->
            maskImageSegmentation?.let {
                val result = it.execute(
                    bitmap
                )
                runOnUiThread {
                    ivOverlay.setImageBitmap(
                        result.first
                    )
                    timestamp?.let { t ->
                        logger.logInfo("fps", System.currentTimeMillis() - t)
                    }
                    timestamp = System.currentTimeMillis()
                }
                if (sendBitmapToStorage) {
                    uploadDataToBacket(result.first)
                    sendBitmapToStorage = false
                }
            }
        }
    }

    private fun uploadDataToBacket(bitmap: Bitmap) {
        val storageRef = Firebase.storage.reference
        val imgRef = storageRef.child("images/${DeviceInfo.uuid}/result-${Date()}.jpg")
        val baos = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, baos)
        val data = baos.toByteArray()
        val uploadTask = imgRef.putBytes(data)
        uploadTask.continueWithTask { task ->
            if (!task.isSuccessful) {
                task.exception?.let {
                    throw it
                }
            }
            imgRef.downloadUrl
        }.addOnCompleteListener { task ->
            val msg =
                if (task.isSuccessful) {
                    copyToClipboard(task.result.toString())
                    "Uploaded Successfully, link to image copied to clipboard"
                } else {
                    "Upload Failed: ${task.exception}"
                }
            Snackbar.make(
                findViewById(R.id.coordinator), msg, Snackbar.LENGTH_LONG
            ).show()
            fab.isEnabled = true
        }
    }

    private fun copyToClipboard(link: String) {
        val clipBoard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clipData = ClipData.newPlainText("link", link)
        clipBoard.setPrimaryClip(clipData)
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