package com.griddynamics.video.conf

import android.opengl.GLSurfaceView
import android.os.Bundle
import android.util.Size
import android.view.LayoutInflater
import android.view.Surface
import android.view.View
import android.view.ViewGroup
import androidx.annotation.IntRange
import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraX
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.UseCase
//import androidx.camera.core.ImageAnalysisConfig
import androidx.camera.core.impl.ImageAnalysisConfig
import androidx.fragment.app.Fragment
import com.griddynamics.video.conf.camera.GlRenderer
import java.util.concurrent.Executors
import androidx.camera.camera2.Camera2Config
import androidx.camera.core.CameraXConfig



private const val EXTRA_CAM = "CAMERA_ID"

class CameraFragment5 : Fragment() {

    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var surfaceView: GLSurfaceView
    private lateinit var renderer: GlRenderer

    companion object {
        fun newInstance(@IntRange(from = 0, to = 1) cameraId: Int) = CameraFragment5()
            .apply {
                arguments = Bundle().apply { putInt(EXTRA_CAM, cameraId) }
            }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        val view = inflater.inflate(R.layout.fragment_camera5, container, false)
        surfaceView = view.findViewById(R.id.glsurface)
        return view
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        renderer = GlRenderer(surfaceView)
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
        val cameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()
        CameraX.initialize(activity!!, getCameraXConfig())
/*        CameraX.bindToLifecycle(
            activity!!,
            cameraSelector,
            imageAnalyzer()
        )*/
    }

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    fun getCameraXConfig(): CameraXConfig {
        return CameraXConfig.Builder.fromConfig(Camera2Config.defaultConfig())
            .setCameraExecutor(executor)
            .build()
    }

    private fun imageAnalyzer(): UseCase {
/*        val analyzerConfig = ImageAnalysisConfig.Builder().apply {
            setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
            setTargetResolution(Size(WIDTH, HEIGHT))
        }.build()*/
        val imageAnalysis = ImageAnalysis.Builder()
            //.setTargetResolution(android.util.Size(480, 640))
            .setTargetRotation(Surface.ROTATION_270)
            .setImageQueueDepth(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
        imageAnalysis?.setAnalyzer(executor, renderer)
        return imageAnalysis
/*
        return ImageAnalysis(analyzerConfig).apply {
            setAnalyzer(executor, renderer)
        }*/
    }

}

const val WIDTH = 640
const val HEIGHT = 640