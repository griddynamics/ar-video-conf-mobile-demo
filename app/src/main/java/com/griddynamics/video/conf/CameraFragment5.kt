package com.griddynamics.video.conf

import android.graphics.PixelFormat
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Size
import android.view.LayoutInflater
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import androidx.annotation.IntRange
import androidx.camera.core.CameraX
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.UseCase
//import androidx.camera.core.ImageAnalysisConfig
import androidx.fragment.app.Fragment
import com.griddynamics.video.conf.camera.GlRenderer
import java.util.concurrent.Executors
import androidx.camera.camera2.Camera2Config
import androidx.camera.core.CameraX.LensFacing.BACK
import androidx.camera.core.CameraX.LensFacing.FRONT
import androidx.camera.core.ImageAnalysisConfig
import com.griddynamics.video.conf.camera.GlRenderer1
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor1




private const val EXTRA_CAM = "CAMERA_ID"

class CameraFragment5 : Fragment() {

    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var surfaceView: GLSurfaceView
    private lateinit var renderer: GlRenderer1
    private lateinit var imageSegmentation: ImageSegmentationModelExecutor1
    private var width: Int = 0
    private var height: Int = 0

    companion object {
        fun newInstance(@IntRange(from = 0, to = 1) cameraId: Int) = CameraFragment5()
            .apply {
                arguments = Bundle().apply { putInt(EXTRA_CAM, cameraId) }
            }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
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
        val view = inflater.inflate(R.layout.fragment_camera5, container, false)
        surfaceView = view.findViewById(R.id.glsurface)
        surfaceView.setZOrderOnTop(true);
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0);
        surfaceView.getHolder().setFormat(PixelFormat.RGBA_8888);
        val metrics = DisplayMetrics()

        activity!!.getWindowManager().getDefaultDisplay().getMetrics(metrics)
/*        width = metrics.widthPixels
        height = metrics.heightPixels*/
        width = 1100
        height = 1100
        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        renderer = GlRenderer1(surfaceView, imageSegmentation, width, height)
        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(2)
        surfaceView.setRenderer(renderer)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY

        start()
    }

    private fun start() {
        surfaceView.post { startCamera() }
    }

    private fun startCamera() {
        // Bind use cases to lifecycle
        // If Android Studio complains about "this" being not a LifecycleOwner
        // try rebuilding the project or updating the appcompat dependency to
        // version 1.1.0 or higher.
        CameraX.bindToLifecycle(
            this,
            imageAnalyzer()
        )
    }


    private fun imageAnalyzer(): UseCase {
        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            setLensFacing(FRONT)
            setTargetResolution(Size(width, height))
        }.build()

        return ImageAnalysis(analyzerConfig).apply {
            setAnalyzer(executor, renderer)
        }
    }

}


/*
const val WIDTH = 900
const val HEIGHT = 900*/
