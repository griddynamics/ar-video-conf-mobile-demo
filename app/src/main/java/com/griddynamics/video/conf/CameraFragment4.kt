package com.griddynamics.video.conf

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.drawable.Drawable
import android.os.Bundle
import android.os.SystemClock
import android.util.DisplayMetrics
import android.util.Log
import android.util.Rational
import android.util.Size
import android.view.LayoutInflater
import android.view.Surface
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import androidx.annotation.IntRange
import androidx.annotation.RequiresPermission
import androidx.camera.core.AspectRatio
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.graphics.scale
import androidx.fragment.app.Fragment
import com.google.common.util.concurrent.ListenableFuture
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor1
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor1.Companion
import com.griddynamics.video.conf.tf.ImageUtils
import com.griddynamics.video.conf.tf.toYUV420Bitmap
import java.io.InputStream
import java.util.concurrent.Executors

private const val EXTRA_CAM = "CAMERA_ID"

class CameraFragment4 : Fragment() {

    companion object {
        fun newInstance(@IntRange(from = 0, to = 1) cameraId: Int) = CameraFragment4()
            .apply {
                arguments = Bundle().apply { putInt(EXTRA_CAM, cameraId) }
            }
    }

/*    private var camera: Camera? = null
    private var imagePreview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>? = null

    private var imageAnalysis: ImageAnalysis? = null
    private var imageSegmentation: ImageSegmentationModelExecutor1? = null
    private val executorAnalyzer = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var modifiedView: SurfaceView
    private var width: Int = 0
    private var height: Int = 0
    private lateinit var background: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraProviderFuture = ProcessCameraProvider.getInstance(requireActivity())
        imageSegmentation = ImageSegmentationModelExecutor1(requireContext(), true)
        imageSegmentation?.initialize()
            ?.addOnSuccessListener { }
            ?.addOnFailureListener { }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_camera4, container, false)
        previewView = view.findViewById(R.id.preview_view)
        modifiedView = view.findViewById(R.id.modified_view)
        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        previewView.post {
            if (ActivityCompat.checkSelfPermission(
                    requireContext(),
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                startCamera()
            }
        }
    }

    @RequiresPermission(value = Manifest.permission.CAMERA)
    private fun startCamera() {
        val cameraId = arguments?.getInt(EXTRA_CAM, CameraSelector.LENS_FACING_FRONT)
            ?: CameraSelector.LENS_FACING_FRONT
        val rotation = previewView.display.rotation
        imagePreview = Preview.Builder()
            .setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setTargetRotation(Surface.ROTATION_270)
            .build()
        val surface = previewView.createSurfaceProvider(null)
        imagePreview?.setSurfaceProvider(surface)

        val metrics = DisplayMetrics().also { modifiedView.display.getRealMetrics(it) }

        imageAnalysis = ImageAnalysis.Builder()
            //.setTargetResolution(android.util.Size(480, 640))
            .setTargetRotation(Surface.ROTATION_270)
            .setImageQueueDepth(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        imageAnalysis?.setAnalyzer(executorAnalyzer, ImageAnalysis.Analyzer { handleFrame(it) })

        imageCapture = ImageCapture.Builder()
            .setTargetRotation(rotation)
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(cameraId)
            .build()

        loadBackground()

        cameraProviderFuture?.addListener(Runnable {
            val cameraProvider = cameraProviderFuture?.get()
            cameraProvider?.unbindAll()
            camera = cameraProvider?.bindToLifecycle(
                viewLifecycleOwner,
                cameraSelector,
                imagePreview,
                imageAnalysis,
                imageCapture
            )
        }, ContextCompat.getMainExecutor(requireActivity()))
    }

    private fun loadBackground() {

        val ims = activity?.assets?.open("beach_photo.jpeg");
        val metrics = DisplayMetrics().also { modifiedView.display.getRealMetrics(it) }
        width = metrics.widthPixels
        height = metrics.heightPixels - 200
        val matrix = Matrix()

        matrix.postRotate(90.0f);

        val tmp = BitmapFactory.decodeStream(ims)
        background =
            Bitmap.createBitmap(tmp, 0, 0, tmp.width, tmp.height, matrix, true).scale(width, height)
    }

    private fun handleFrame(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toYUV420Bitmap(90)
        imageSegmentation?.executeAsync(bitmap)
            ?.addOnSuccessListener { result ->
                result?.bitmapResult ?: return@addOnSuccessListener
                val canvas = modifiedView.holder.lockCanvas() ?: return@addOnSuccessListener

                canvas.drawBitmap(background, canvas.matrix, null)
                canvas.drawBitmap(result?.bitmapResult.scale(width, height), canvas.matrix, null)
                modifiedView.holder.unlockCanvasAndPost(canvas)
                bitmap.recycle()
                result.recycle()
                imageProxy.close()
            }
            ?.addOnFailureListener {
                bitmap.recycle()
                imageProxy.close()
            }
    }

    override fun onDestroy() {
        imageSegmentation?.destroy()
        super.onDestroy()
    }*/

}