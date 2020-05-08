package com.griddynamics.video.conf;

import android.content.Context;
import android.graphics.SurfaceTexture;
import android.media.AudioManager;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import com.google.mediapipe.components.CameraHelper;
import com.google.mediapipe.components.CameraXPreviewHelper;
import com.google.mediapipe.components.ExternalTextureConverter;
import com.google.mediapipe.components.FrameProcessor;
import com.google.mediapipe.components.PermissionHelper;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmark;
import com.google.mediapipe.formats.proto.LandmarkProto.NormalizedLandmarkList;
import com.google.mediapipe.framework.AndroidAssetUtil;
import com.google.mediapipe.framework.PacketGetter;
import com.google.mediapipe.glutil.EglManager;
import com.griddynamics.video.conf.calculator.HandGesture;
import com.griddynamics.video.conf.calculator.HandGestureCalculator;
import com.griddynamics.video.conf.texture.converter.PalmTextureConverter;

import java.util.List;

/** Main activity of MediaPipe example apps. */
public class PalmDetectionActivity extends AppCompatActivity {
    private static final String TAG = "PalmDetectionActivity";
    private static final String BINARY_GRAPH_NAME = "multihandtrackinggpu.binarypb";
    private static final String INPUT_VIDEO_STREAM_NAME = "input_video";
    private static final String OUTPUT_VIDEO_STREAM_NAME = "output_video";
    private static final String OUTPUT_LANDMARKS_STREAM_NAME = "multi_hand_landmarks";
    private static final CameraHelper.CameraFacing CAMERA_FACING = CameraHelper.CameraFacing.FRONT;
    // Flips the camera-preview frames vertically before sending them into FrameProcessor to be
    // processed in a MediaPipe graph, and flips the processed frames back when they are displayed.
    // This is needed because OpenGL represents images assuming the image origin is at the bottom-left
    // corner, whereas MediaPipe in general assumes the image origin is at top-left.
    private static final boolean FLIP_FRAMES_VERTICALLY = true;
    static {
        // Load all native libraries needed by the app.
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }
    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private SurfaceTexture previewFrameTexture;
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private SurfaceView previewDisplayView;
    // Creates and manages an {@link EGLContext}.
    private EglManager eglManager;
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private FrameProcessor processor;
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private PalmTextureConverter converter;
    // Handles camera access via the {@link CameraX} Jetpack support library.
    private CameraXPreviewHelper cameraHelper;

    private AudioManager audioManager = null;
    private TextView txtGesture = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_palm_detection);
        previewDisplayView = new SurfaceView(this);
        setupPreviewDisplayView();
        txtGesture = findViewById(R.id.txtGesture);
        txtGesture.bringToFront();
        // Initialize asset manager so that MediaPipe native libraries can access the app assets, e.g.,
        // binary graphs.
        AndroidAssetUtil.initializeNativeAssetManager(this);
        eglManager = new EglManager(null);
        processor =
                new FrameProcessor(
                        this,
                        eglManager.getNativeContext(),
                        BINARY_GRAPH_NAME,
                        INPUT_VIDEO_STREAM_NAME,
                        OUTPUT_VIDEO_STREAM_NAME);
        processor.getVideoSurfaceOutput().setFlipY(FLIP_FRAMES_VERTICALLY);
        processor.addPacketCallback(
                OUTPUT_LANDMARKS_STREAM_NAME,
                (packet) -> {
                    Log.d(TAG, "Received multi-hand landmarks packet.");
                    List<NormalizedLandmarkList> multiHandLandmarks =
                            PacketGetter.getProtoVector(packet, NormalizedLandmarkList.parser());
                    HandGestureCalculator calculator = new HandGestureCalculator();
                    HandGesture gesture = calculator.detectGesture(multiHandLandmarks);
                    if (gesture == HandGesture.FIST) {
                        Log.d("HandGesture", "FIST");
                        audioManager.adjustVolume(AudioManager.ADJUST_LOWER, AudioManager.FLAG_PLAY_SOUND);
                    }

                    if (gesture == HandGesture.OPEN_HAND) {
                        Log.d("HandGesture", "OPEN_HAND");
                        audioManager.adjustVolume(AudioManager.ADJUST_RAISE, AudioManager.FLAG_PLAY_SOUND);
                    }
                    runOnUiThread(() -> txtGesture.setText(gesture.toString()));


                });
        PermissionHelper.checkAndRequestCameraPermissions(this);
        audioManager = (AudioManager) getApplicationContext().getSystemService(Context.AUDIO_SERVICE);
    }
    @Override
    protected void onResume() {
        super.onResume();
        converter = new PalmTextureConverter(eglManager.getContext());
        converter.setFlipY(FLIP_FRAMES_VERTICALLY);
        converter.setConsumer(processor);
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera();
        }
    }
    @Override
    protected void onPause() {
        super.onPause();
        converter.close();
    }
    @Override
    public void onRequestPermissionsResult(
            int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults);
    }
    private void setupPreviewDisplayView() {
        previewDisplayView.setVisibility(View.GONE);
        ViewGroup viewGroup = findViewById(R.id.preview_display_layout);
        viewGroup.addView(previewDisplayView);
        previewDisplayView
                .getHolder()
                .addCallback(
                        new SurfaceHolder.Callback() {
                            @Override
                            public void surfaceCreated(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(holder.getSurface());
                            }
                            @Override
                            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
                                // (Re-)Compute the ideal size of the camera-preview display (the area that the
                                // camera-preview frames get rendered onto, potentially with scaling and rotation)
                                // based on the size of the SurfaceView that contains the display.
                                Size viewSize = new Size(width, height);
                                Size displaySize = cameraHelper.computeDisplaySizeFromViewSize(viewSize);
                                // Connect the converter to the camera-preview frames as its input (via
                                // previewFrameTexture), and configure the output width and height as the computed
                                // display size.
                                converter.setSurfaceTextureAndAttachToGLContext(
                                        previewFrameTexture, displaySize.getWidth(), displaySize.getHeight());
                            }
                            @Override
                            public void surfaceDestroyed(SurfaceHolder holder) {
                                processor.getVideoSurfaceOutput().setSurface(null);
                            }
                        });
    }
    private void startCamera() {
        cameraHelper = new CameraXPreviewHelper();
        cameraHelper.setOnCameraStartedListener(
                surfaceTexture -> {
                    previewFrameTexture = surfaceTexture;
                    // Make the display view visible to start showing the preview. This triggers the
                    // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
                    previewDisplayView.setVisibility(View.VISIBLE);
                    txtGesture.bringToFront();
                });
        cameraHelper.startCamera(this, CAMERA_FACING, /*surfaceTexture=*/ null);
    }
}