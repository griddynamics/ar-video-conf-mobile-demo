package com.griddynamics.video.conf

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.LayoutInflater
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import androidx.annotation.IntRange
import androidx.annotation.RequiresPermission
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.google.common.util.concurrent.ListenableFuture
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor
import com.griddynamics.video.conf.tf.toYUV420Bitmap
import java.util.concurrent.Executors

private const val EXTRA_CAM = "CAMERA_ID"

class CameraFragment4 : Fragment() {

    companion object {
        fun newInstance(@IntRange(from = 0, to = 1) cameraId: Int) = CameraFragment4()
            .apply {
                arguments = Bundle().apply { putInt(EXTRA_CAM, cameraId) }
            }
    }

    private var camera: Camera? = null
    private var imagePreview: Preview? = null
    private var imageCapture: ImageCapture? = null
    private var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>? = null

    private var imageAnalysis: ImageAnalysis? = null
    private var imageSegmentation: ImageSegmentationModelExecutor? = null
    private val executorAnalyzer = Executors.newSingleThreadExecutor()

    private lateinit var previewView: PreviewView
    private lateinit var modifiedView: SurfaceView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraProviderFuture = ProcessCameraProvider.getInstance(requireActivity())
        imageSegmentation = ImageSegmentationModelExecutor(requireContext(), true)
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
            .setTargetAspectRatio(AspectRatio.RATIO_16_9)
            .setTargetRotation(rotation)
            .build()
        val surface = previewView.createSurfaceProvider(null)
        imagePreview?.setSurfaceProvider(surface)

        imageAnalysis = ImageAnalysis.Builder()
            /*.setTargetResolution(android.util.Size(480, 640))*/
            .setTargetRotation(rotation)
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

    private fun handleFrame(imageProxy: ImageProxy) {
        val bitmap = imageProxy.toYUV420Bitmap(90)
        imageSegmentation?.executeAsync(bitmap)
            ?.addOnSuccessListener { result ->
                val resultBitmap = result?.bitmapResult ?: return@addOnSuccessListener
                val canvas = modifiedView.holder.lockCanvas() ?: return@addOnSuccessListener
                canvas.drawBitmap(resultBitmap, canvas.matrix, null)
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
    }

}