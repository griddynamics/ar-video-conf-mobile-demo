package com.griddynamics.video.conf

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.*
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
import androidx.core.graphics.scale
import com.griddynamics.video.conf.utils.BuildUtils.Companion.isEmulator
import com.griddynamics.video.conf.utils.getTransformationMatrixContain
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
        private const val TF_OD_API_INPUT_SIZE = 256
        private const val TF_OD_API_IS_QUANTIZED = false
        private const val TF_OD_API_MODEL_FILE = "palm_detection_without_custom_op.tflite"
        private const val TF_OD_API_LABELS_FILE =
            "file:///android_asset/palm_detection_labelmap.txt"
    }

    private lateinit var tracker: MultiBoxTracker
    private val coroutineScope = CoroutineScope(Job())
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private val i256 = 256
    private val deepLabLite: DeepLabLite = DeepLabLite()
    private lateinit var maskImageSegmentation: ImageSegmentation
    private lateinit var handModel: Classifier
    private val imageSegmentationFactory = ImageSegmentationFactory()
    private var useDeeplab = false
    private var timestamp = System.currentTimeMillis()

    private val cropToFrameTransform = Matrix()
    private var frameToCropTransform = Matrix()


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
            maskImageSegmentation = imageSegmentationFactory.provideCustom(applicationContext)
            handModel = TFLiteObjectDetectionAPIModel.create(
                assets,
                TF_OD_API_MODEL_FILE,
                TF_OD_API_LABELS_FILE,
                TF_OD_API_INPUT_SIZE,
                TF_OD_API_IS_QUANTIZED
            )

            frameToCropTransform = getTransformationMatrixContain(
                1262,
                720,
                256,
                256,
                90,
                true
            )
            frameToCropTransform.invert(cropToFrameTransform)
            deepLabLite.initialize(applicationContext)
            runOnUiThread {
                btnGD.callOnClick()
            }
        }
        tracker = MultiBoxTracker(this)
        gestureOverlay.addCallback { canvas ->
            tracker.draw(canvas)
        }

        tracker.setFrameConfiguration(1262, 720, 90)
    }

    private fun setupButtons() {
        btnNothing.setOnClickListener {
            btnNothing.scaleX = 1.4f
            btnNothing.scaleY = 1.4f
            btnBlur.scaleX = 1f
            btnBlur.scaleY = 1f
            btnGD.scaleX = 1f
            btnGD.scaleY = 1f
            btnHand.scaleX = 1f
            btnHand.scaleY = 1f

            ivOverlay.visibility = View.INVISIBLE
            ivBack.visibility = View.INVISIBLE
            viewFinder.visibility = View.VISIBLE
            maskImageSegmentation.applyBlur = false
            useDeeplab = false
        }
        btnGD.setOnClickListener {
            btnNothing.scaleX = 1f
            btnNothing.scaleY = 1f
            btnBlur.scaleX = 1f
            btnBlur.scaleY = 1f
            btnGD.scaleX = 1.4f
            btnGD.scaleY = 1.4f
            btnHand.scaleX = 1f
            btnHand.scaleY = 1f

            ivOverlay.visibility = View.VISIBLE
            ivBack.visibility = View.VISIBLE
            viewFinder.visibility = View.INVISIBLE
            maskImageSegmentation.applyBlur = false
            useDeeplab = false
        }
        btnBlur.setOnClickListener {
            btnNothing.scaleX = 1f
            btnNothing.scaleY = 1f
            btnBlur.scaleX = 1.4f
            btnBlur.scaleY = 1.4f
            btnGD.scaleX = 1f
            btnGD.scaleY = 1f
            btnHand.scaleX = 1f
            btnHand.scaleY = 1f

            ivOverlay.visibility = View.VISIBLE
            ivBack.visibility = View.INVISIBLE
            viewFinder.visibility = View.INVISIBLE
            maskImageSegmentation.applyBlur = true
            useDeeplab = false
        }
        btnHand.setOnClickListener {
            btnNothing.scaleX = 1f
            btnNothing.scaleY = 1f
            btnBlur.scaleX = 1f
            btnBlur.scaleY = 1f
            btnGD.scaleX = 1f
            btnGD.scaleY = 1f
            btnHand.scaleX = 1.4f
            btnHand.scaleY = 1.4f

            ivOverlay.visibility = View.INVISIBLE
            ivBack.visibility = View.INVISIBLE
            viewFinder.visibility = View.VISIBLE
            maskImageSegmentation.applyBlur = false
            useDeeplab = true
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
            if (useDeeplab) {
                val tmp = bitmap.scale(256,256)
                val results: List<Classifier.Recognition> =
                    handModel.recognizeImage(tmp, tmp)


                val cropCopyBitmap = Bitmap.createBitmap(tmp)
                val canvas = Canvas(cropCopyBitmap)
                val paint = Paint()
                paint.color = Color.RED
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = 2.0f

                val mappedRecognitions: MutableList<Classifier.Recognition> = LinkedList()

                for (result in results) {
                    val location = result.location
                    if (location != null && result.confidence >= 0) {
                        canvas.drawRect(location, paint)

                        cropToFrameTransform.mapRect(location)
                        result.location = location
                        mappedRecognitions.add(result)
                    }
                }
                runOnUiThread {
                    tracker.trackResults(mappedRecognitions, System.currentTimeMillis())
                    gestureOverlay.postInvalidate()
                }
            } else {
                val result = maskImageSegmentation.execute(
                    bitmap
                )
                runOnUiThread {
                    //     val diff = System.currentTimeMillis() - timestamp
                    //     if (diff > 0)
                    //         tvInfo.text = (1000f / diff).toString() + "fps"
                    ivOverlay.setImageBitmap(
                        result
                    )
                    timestamp = System.currentTimeMillis()
                }
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