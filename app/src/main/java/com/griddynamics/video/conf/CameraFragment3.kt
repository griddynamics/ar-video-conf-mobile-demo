package com.griddynamics.video.conf

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.hardware.camera2.CameraDevice.StateCallback
import android.media.Image
import android.media.ImageReader
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.*
import androidx.fragment.app.Fragment
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor
import com.griddynamics.video.conf.tf.toJPEGBitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.Closeable
import java.io.File
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class CameraFragment3 : Fragment() {
    private var cameraInitialized: Boolean = false
    private val fragmentJob = SupervisorJob()
    private val fragmentScope = CoroutineScope(Dispatchers.Main + fragmentJob)

    // interface to interact with the hosting activity
    interface OnCaptureFinished {
        fun onCaptureFinished(file: File)
    }

    private var cameraFacing = CameraCharacteristics.LENS_FACING_FRONT
    private var cameraId = "0"

    private var imageSegmentation: ImageSegmentationModelExecutor? = null

    /**
     * define the aspect ratio that will be used by the camera
     */
    private val aspectRatio = Size(4, 3)

    /**
     * Max preview width that is guaranteed by Camera2 API
     */
    private val MAX_PREVIEW_WIDTH = 1080

    internal lateinit var callback: OnCaptureFinished

    /** Detects, characterizes, and connects to a CameraDevice (used for all camera operations) */
    private val cameraManager: CameraManager by lazy {
        val context = requireContext().applicationContext
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager
    }

    /** [CameraCharacteristics] corresponding to the provided Camera ID */
    private lateinit var characteristics: CameraCharacteristics

    /** Readers used as buffers for camera still shots */
    private lateinit var imageReader: ImageReader

    /** [HandlerThread] where all camera operations run */
    private val cameraThread = HandlerThread("CameraThread").apply { start() }

    /** [Handler] corresponding to [cameraThread] */
    private val cameraHandler = Handler(cameraThread.looper)

    /** [HandlerThread] where all buffer reading operations run */
    private val imageReaderThread = HandlerThread("imageReaderThread").apply { start() }

    /** [Handler] corresponding to [imageReaderThread] */
    private val imageReaderHandler = Handler(imageReaderThread.looper)

    /** Where the camera preview is displayed */
    private lateinit var viewFinder: SurfaceView
    private lateinit var encodedVideo: SurfaceView

    /** Internal reference to the ongoing [CameraCaptureSession] configured with our parameters */
    private lateinit var session: CameraCaptureSession

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        imageSegmentation = ImageSegmentationModelExecutor(requireContext(), true)
        imageSegmentation?.initialize()
            ?.addOnSuccessListener { Log.d(TAG, "Image segmentation success") }
            ?.addOnFailureListener { Log.d(TAG, "Image segmentation error: $it") }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val view = inflater.inflate(R.layout.fragment_camera1, container, false)
        viewFinder = view.findViewById(R.id.autoFit)
        encodedVideo = view.findViewById(R.id.autoFit1)
        return view
    }

    @SuppressLint("MissingPermission")
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        //viewFinder = view as AutoFitSurfaceView

        cameraId = getCameraId()
        characteristics = cameraManager.getCameraCharacteristics(cameraId)
        val size = characteristics.get(
            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP
        )!!
            .getOutputSizes(ImageFormat.JPEG).first {
                it.width <= MAX_PREVIEW_WIDTH && verifyAspectRatio(it.width, it.height, aspectRatio)
            }
        imageReader = ImageReader.newInstance(
            size.width,
            size.height,
            ImageFormat.JPEG,
            IMAGE_BUFFER_SIZE
        )
        imageReader.setOnImageAvailableListener({ drawCanvas(it) }, imageReaderHandler)

        viewFinder.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceDestroyed(holder: SurfaceHolder) = Unit

            override fun surfaceChanged(
                holder: SurfaceHolder,
                format: Int,
                width: Int,
                height: Int
            ) = Unit

            override fun surfaceCreated(holder: SurfaceHolder) {
                // Selects appropriate preview size and configures view finder
                val previewSize = getPreviewOutputSize(
                    viewFinder.display, characteristics, SurfaceHolder::class.java, aspectRatio
                )
                Log.d(TAG, "View finder size: ${viewFinder.width} x ${viewFinder.height}")
                Log.d(TAG, "Selected preview size: $previewSize")
                viewFinder.holder.setFixedSize(previewSize.width, previewSize.height)
                (viewFinder as AutoFitSurfaceView).setAspectRatio(
                    previewSize.width,
                    previewSize.height
                )

                // To ensure that size is set, initialize camera in the view's thread
                view.post { initializeCamera() }
            }
        })

/*        // Used to rotate the output media to match device orientation
        relativeOrientation = OrientationLiveData(requireContext(), characteristics).apply {
            observe(
                this@CameraFragment,
                Observer {
                        orientation ->
                    Log.d(TAG, "Orientation changed: $orientation")
                }
            )
        }*/
    }

    private var isProcessing = false

    private fun drawCanvas(reader: ImageReader) {
        reader.acquireLatestImage().use {
            if (isProcessing) {
                return@use
            }
            isProcessing = true
            val bitmap = it?.toJPEGBitmap(90) ?: return@use
            imageSegmentation?.executeAsync(bitmap)
                ?.addOnSuccessListener { result ->
                    // take necessary bitmap
                    val resultBitmap = result?.bitmapResult ?: return@addOnSuccessListener
                    // draw bitmaps
                    val canvas = encodedVideo.holder.lockCanvas() ?: return@addOnSuccessListener
                    canvas.drawBitmap(resultBitmap, canvas.matrix, null)
                    encodedVideo.holder.unlockCanvasAndPost(canvas)
                    // recycle bitmaps
                    bitmap.recycle()
                    result.recycle()
                    // ready for next frame
                    isProcessing = false
                }
                ?.addOnFailureListener {
                    // recycle bitmaps
                    bitmap.recycle()
                    // ready for next frame
                    isProcessing = false
                }
        }
/*        val rgbFrameBitmap = Utl.onImageAvailable(it)
        rgbFrameBitmap?.let {
            val canvas = encodedVideo.holder.lockCanvas()
            canvas.drawBitmap(rgbFrameBitmap, canvas.matrix, null)
            encodedVideo.holder.unlockCanvasAndPost(canvas)


            //canvas.drawBitmap(rgbFrameBitmap, canvas.matrix, null)
        }*/
/*        this.activity?.runOnUiThread {
            Log.d(TAG, "drawCanvas")
            val img = it.acquireLatestImage()
            if(img != null){
                img.close()
            }

        }*/

    }

    /**
     * Begin all camera operations in a coroutine in the main thread. This function:
     * - Opens the camera
     * - Configures the camera session
     * - Starts the preview by dispatching a repeating capture request
     * - Sets up the image capture listeners
     */
    private fun initializeCamera() = fragmentScope.launch(Dispatchers.Main) {
        val camera = openCamera()
        session = startCaptureSession(camera)
        cameraInitialized = true
        val captureRequest = camera.createCaptureRequest(
            CameraDevice.TEMPLATE_RECORD
        )
            .apply {
                addTarget(viewFinder.holder.surface)
                addTarget(imageReader.surface)
            }


        // This will keep sending the capture request as frequently as possible until the
        // session is torn down or session.stopRepeating() is called
        session.setRepeatingRequest(captureRequest.build(), null, cameraHandler)
    }

    /** Opens the camera and returns the opened device (as the result of the suspend coroutine) */
    @SuppressLint("MissingPermission")
    private suspend fun openCamera(): CameraDevice = suspendCoroutine { cont ->
        val cameraId = getCameraId()
        cameraManager.openCamera(
            cameraId,
            object : StateCallback() {
                override fun onDisconnected(device: CameraDevice) {
                    Log.w(TAG, "Camera $cameraId has been disconnected")
                    requireActivity().finish()
                }

                override fun onError(device: CameraDevice, error: Int) {
                    val exc = RuntimeException("Camera $cameraId open error: $error")
                    Log.e(TAG, exc.message, exc)
                    cont.resumeWithException(exc)
                }

                override fun onOpened(device: CameraDevice) = cont.resume(device)
            },
            cameraHandler
        )
    }

    private fun getCameraId(): String {
        // Get list of all compatible cameras
        val cameraIds = cameraManager.cameraIdList.filter {
            val characteristics = cameraManager.getCameraCharacteristics(it)
            val capabilities = characteristics.get(
                CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES
            )
            capabilities?.contains(
                CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_BACKWARD_COMPATIBLE
            ) ?: false
        }

        // Iterate over the list of cameras and return the one that is facing the selected direction
        cameraIds.forEach { id ->
            val characteristics = cameraManager.getCameraCharacteristics(id)
            if (cameraFacing == characteristics.get(CameraCharacteristics.LENS_FACING)) {
                return id
            }
        }
        return cameraId.first().toString()
    }

    /**
     * Starts a [CameraCaptureSession] and returns the configured session (as the result of the
     * suspend coroutine
     */
    private suspend fun startCaptureSession(device: CameraDevice):
            CameraCaptureSession = suspendCoroutine { cont ->

        // Create list of Surfaces where the camera will output frames
        val targets: MutableList<Surface> =
            arrayOf(viewFinder.holder.surface, imageReader.surface).toMutableList()
        //viewFinder.setOnI

        // Create a capture session using the predefined targets; this also involves defining the
        // session state callback to be notified of when the session is ready
        device.createCaptureSession(
            targets,
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigureFailed(session: CameraCaptureSession) {
                    val exc = RuntimeException(
                        "Camera ${device.id} session configuration failed, see log for details"
                    )
                    Log.e(TAG, exc.message, exc)
                    cont.resumeWithException(exc)
                }

                override fun onConfigured(session: CameraCaptureSession) = cont.resume(session)
            },
            cameraHandler
        )
    }

    override fun onStop() {
        super.onStop()
        try {
            session.close()
            session.device.close()
        } catch (exc: Throwable) {
            Log.e(TAG, "Error closing camera", exc)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        imageSegmentation?.destroy()
        fragmentJob.cancel()
        cameraThread.quitSafely()
        imageReaderThread.quitSafely()
    }

    /**
    Keeping a reference to the activity to make communication between it and this fragment
    easier.
     */
    override fun onAttach(context: Context) {
        super.onAttach(context)
        //callback = context as OnCaptureFinished
    }

    fun setFacingCamera(lensFacing: Int) {
        cameraFacing = lensFacing
    }


    companion object {
        private val TAG = CameraFragment3::class.java.simpleName

        /** Maximum number of images that will be held in the reader's buffer */
        private const val IMAGE_BUFFER_SIZE: Int = 3

        /** Maximum time allowed to wait for the result of an image capture */
        private const val IMAGE_CAPTURE_TIMEOUT_MILLIS: Long = 5000

        /** Helper data class used to hold capture metadata results with their associated image */
        data class CombinedCaptureResult(
            val image: Image,
            val metadata: CaptureResult,
            val orientation: Int,
            val format: Int
        ) : Closeable {
            override fun close() = image.close()
        }

        fun newInstance(): CameraFragment3 =
            CameraFragment3()
    }

}