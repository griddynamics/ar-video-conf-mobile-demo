package com.griddynamics.video.conf

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.bumptech.glide.Glide
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutorCustomStatic
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutorStatic
import com.griddynamics.video.conf.tf.ModelExecutionResult
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.Executors


private const val REQUEST_CODE_PERMISSIONS = 10
class StaticAnalyzerActivity : AppCompatActivity(), CameraFragment.OnCaptureFinished {



    private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

    private lateinit var cameraFragment: CameraFragment

    private val completableJob = Job()
    private val coroutineScope = CoroutineScope(completableJob)
    private val inferenceThread = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private lateinit var viewFinder: FrameLayout
    private lateinit var resultImageView: ImageView
    private lateinit var originalImageView: ImageView
    private lateinit var maskImageView: ImageView
    private lateinit var resultImageViewCustom: ImageView
    private lateinit var originalImageViewCustom: ImageView
    private lateinit var maskImageViewCustom: ImageView
    private lateinit var captureButton: ImageButton
    private lateinit var txtInitializing: TextView
    private lateinit var imageSegmentation: ImageSegmentationModelExecutorStatic
    private lateinit var imageSegmentationCustomStatic: ImageSegmentationModelExecutorCustomStatic

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_static_analyzer)
        viewFinder = findViewById(R.id.view_finder)
        resultImageView = findViewById(R.id.result_imageview)
        originalImageView = findViewById(R.id.original_imageview)
        maskImageView = findViewById(R.id.mask_imageview)

        resultImageViewCustom = findViewById(R.id.result_imageview_custom)
        originalImageViewCustom = findViewById(R.id.original_imageview_custom)
        maskImageViewCustom = findViewById(R.id.mask_imageview_custom)
        txtInitializing = findViewById(R.id.txtInitializing)

        captureButton = findViewById(R.id.capture_button)
        captureButton.visibility = View.GONE
        if (allPermissionsGranted()) {
            addCameraFragment()
        } else {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }

        setupControls()
    }

    override fun onResume() {
        super.onResume()
        imageSegmentation = ImageSegmentationModelExecutorStatic(this, false)
        imageSegmentation?.initialize()
            ?.addOnSuccessListener {
                txtInitializing.visibility = View.GONE
                captureButton.visibility = View.VISIBLE
            }
            ?.addOnFailureListener { l ->
                run {
                    txtInitializing.text = l.message
                    Log.d("ImageSegmentationModel", "Error: $l")
                }
            }

        imageSegmentationCustomStatic = ImageSegmentationModelExecutorCustomStatic(this, false)
        imageSegmentationCustomStatic?.initialize()
            ?.addOnSuccessListener {
                captureButton.visibility = View.VISIBLE
            }
            ?.addOnFailureListener { l ->
                run {
                    Log.d("ImageSegmentationModel", "Error: $l")
                }
            }

        imageSegmentationCustomStatic
    }

    private fun setupControls() {
        captureButton.setOnClickListener {
            it.clearAnimation()
            cameraFragment.takePicture()
        }
        addCameraFragment()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                addCameraFragment()
                viewFinder.post { setupControls() }
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

    private fun addCameraFragment() {
        cameraFragment = CameraFragment.newInstance()
        supportFragmentManager.popBackStack()
        supportFragmentManager.beginTransaction()
            .replace(R.id.view_finder, cameraFragment)
            .commit()
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it
        ) == PackageManager.PERMISSION_GRANTED
    }

    override fun onCaptureFinished(file: File) {

    }

    override fun onCaptureFinishedPic(bmp: Bitmap) {
        coroutineScope.launch(inferenceThread) {
            val matrix = Matrix()
            //matrix.postRotate(180.0f)

            val largeIcon =
                BitmapFactory.decodeResource(resources, R.drawable.img)
           // val bm = Bitmap.createBitmap(largeIcon, 0, 0, largeIcon.width, largeIcon.height, matrix, true)
            val result = imageSegmentation.execute(largeIcon)
            updateUIWithResults(result)
            val resultStatic = imageSegmentationCustomStatic.execute(largeIcon, this@StaticAnalyzerActivity)
            updateCustomUIWithResults(resultStatic)
        }
    }

    private fun updateUIWithResults(modelExecutionResult: ModelExecutionResult) {
        runOnUiThread {
            setImageView(resultImageView, modelExecutionResult.bitmapResult)
            setImageView(originalImageView, modelExecutionResult.bitmapOriginal)
            setImageView(maskImageView, modelExecutionResult.bitmapMaskOnly)
        }
    }

    private fun updateCustomUIWithResults(modelExecutionResult: ModelExecutionResult) {
        runOnUiThread {
            setImageView(resultImageViewCustom, modelExecutionResult.bitmapResult)
            setImageView(originalImageViewCustom, modelExecutionResult.bitmapOriginal)
            setImageView(maskImageViewCustom, modelExecutionResult.bitmapMaskOnly)
        }
    }

    private fun setImageView(imageView: ImageView, image: Bitmap) {
        Glide.with(baseContext)
            .load(image)
            .override(512, 512)
            .fitCenter()
            .into(imageView)
    }
}
