package com.griddynamics.video.conf

import android.content.Intent
import android.graphics.PixelFormat
import android.opengl.GLSurfaceView
import android.os.Build
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import androidx.annotation.IntRange
import androidx.camera.core.CameraX
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.UseCase
//import androidx.camera.core.ImageAnalysisConfig
import androidx.fragment.app.Fragment
import java.util.concurrent.Executors
import androidx.camera.core.CameraX.LensFacing.FRONT
import androidx.camera.core.ImageAnalysisConfig
import com.griddynamics.video.conf.camera.GlRenderer1
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor1
import android.graphics.BitmapFactory
import android.graphics.Bitmap
import android.text.TextUtils
import com.griddynamics.video.conf.camera.GlRendererCustom
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutorCustom
import java.io.File




private const val EXTRA_CAM = "CAMERA_ID"

class CameraFragment5 : Fragment() {

    private val REQUEST_REQUIRED_PERMISSION = 0x01
    private val REQUEST_PICK_IMAGE = 0x02
    private val executor = Executors.newSingleThreadExecutor()
    private lateinit var surfaceView: GLSurfaceView
    private lateinit var renderer: GlRendererCustom
    private lateinit var imageSegmentation: ImageSegmentationModelExecutorCustom
    private lateinit var imgBackground: ImageView
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
        imageSegmentation = ImageSegmentationModelExecutorCustom(requireContext(), true)
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
        surfaceView.setZOrderOnTop(true)
        surfaceView.setEGLConfigChooser(8, 8, 8, 8, 16, 0)
        surfaceView.holder.setFormat(PixelFormat.RGBA_8888)
        imgBackground = view.findViewById(R.id.imgBackground)
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
        renderer = GlRendererCustom(surfaceView, imageSegmentation, width, height)
        surfaceView.preserveEGLContextOnPause = true
        surfaceView.setEGLContextClientVersion(2)
        surfaceView.setRenderer(renderer)
        surfaceView.renderMode = GLSurfaceView.RENDERMODE_WHEN_DIRTY
        loadBackground()
    }

    private fun loadBackground() {
        val intent: Intent

        if (Build.VERSION.SDK_INT >= 19) {
            intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, false)
            intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION)
        } else {
            intent = Intent(Intent.ACTION_GET_CONTENT)
        }

        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        intent.type = "image/*"

        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)

        startActivityForResult(intent, REQUEST_PICK_IMAGE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        if (requestCode == REQUEST_PICK_IMAGE && data != null) {
            val pickedImageUri = data.data

            if (pickedImageUri != null) {
                if (Build.VERSION.SDK_INT >= 19) {
                    val takeFlags = data.flags and Intent.FLAG_GRANT_READ_URI_PERMISSION
                    activity!!.contentResolver
                        .takePersistableUriPermission(pickedImageUri, takeFlags)
                }
                val filePath = FilePickUtils.getPath(context, pickedImageUri)
                val options = BitmapFactory.Options()
                options.inJustDecodeBounds = true
                val myBitmap = Utl.compressBitmap(filePath)

                imgBackground.setImageBitmap(myBitmap)
            } else {
                imgBackground.setImageResource(R.drawable.beach_photo)
            }
        }
        startProcess()
    }

    private fun startProcess() {
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
