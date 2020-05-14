package com.griddynamics.video.conf.camera;

import android.app.Activity;
import android.graphics.SurfaceTexture;
import android.opengl.GLSurfaceView;
import android.util.Log;
import android.util.Size;
import androidx.annotation.Nullable;
import androidx.camera.core.CameraX;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.camera.core.UseCase;
import androidx.lifecycle.LifecycleOwner;
import androidx.camera.core.PreviewConfig.Builder;
import com.google.mediapipe.components.CameraHelper;
import com.griddynamics.video.conf.tf.ImageSegmentationModelExecutor1;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VideoActivityCameraHelper extends CameraHelper{
    private static final String TAG = "VideoActivityCameraHelper";
    private Preview preview;
    private Size frameSize;
    private int frameRotation;
    private ExecutorService executor = Executors.newSingleThreadExecutor();
    private VideoRenderer renderer;
    GLSurfaceView surfaceView;

    public VideoActivityCameraHelper(GLSurfaceView surfaceView) {
        this.surfaceView = surfaceView;
    }

    public void startCamera(Activity context, CameraFacing cameraFacing, SurfaceTexture surfaceTexture) {
        CameraX.LensFacing cameraLensFacing = cameraFacing == CameraHelper.CameraFacing.FRONT ? CameraX.LensFacing.FRONT : CameraX.LensFacing.BACK;
        PreviewConfig previewConfig = (new Builder()).setLensFacing(cameraLensFacing).build();
        this.preview = new Preview(previewConfig);
        this.preview.setOnPreviewOutputUpdateListener((previewOutput) -> {
            if (!previewOutput.getTextureSize().equals(this.frameSize)) {
                this.frameSize = previewOutput.getTextureSize();
                this.frameRotation = previewOutput.getRotationDegrees();
                if (this.frameSize.getWidth() == 0 || this.frameSize.getHeight() == 0) {
                    Log.d("VideoActivityCameraHelp", "Invalid frameSize.");
                    return;
                }
            }

            if (this.onCameraStartedListener != null) {
                this.onCameraStartedListener.onCameraStarted(previewOutput.getSurfaceTexture());
            }

        });
        CameraX.bindToLifecycle((LifecycleOwner)context, new UseCase[]{this.preview, imageAnalyzerUseCase(context)});
    }

    private UseCase imageAnalyzerUseCase(Activity context) {
        ImageSegmentationModelExecutor1 imageSegmentation = new ImageSegmentationModelExecutor1(context, true);
        imageSegmentation.initialize();
        renderer = new VideoRenderer(surfaceView, imageSegmentation, 1100, 1100);
        ImageAnalysisConfig analyzerConfig = new ImageAnalysisConfig.Builder().
            setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE).
            setLensFacing(CameraX.LensFacing.FRONT).
            setTargetResolution(new Size(1100, 1100))
        .build();
        ImageAnalysis analysis = new ImageAnalysis(analyzerConfig);
        analysis.setAnalyzer(executor, renderer);
        surfaceView.setPreserveEGLContextOnPause(true);
        surfaceView.setEGLContextClientVersion(2);
        surfaceView.setRenderer(renderer);
        surfaceView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);


        return analysis;
    }

    public Size computeDisplaySizeFromViewSize(Size viewSize) {
        if (viewSize != null && this.frameSize != null) {
            float frameAspectRatio = this.frameRotation != 90 && this.frameRotation != 270 ? (float)this.frameSize.getWidth() / (float)this.frameSize.getHeight() : (float)this.frameSize.getHeight() / (float)this.frameSize.getWidth();
            float viewAspectRatio = (float)viewSize.getWidth() / (float)viewSize.getHeight();
            int scaledWidth;
            int scaledHeight;
            if (frameAspectRatio < viewAspectRatio) {
                scaledWidth = viewSize.getWidth();
                scaledHeight = Math.round((float)viewSize.getWidth() / frameAspectRatio);
            } else {
                scaledHeight = viewSize.getHeight();
                scaledWidth = Math.round((float)viewSize.getHeight() * frameAspectRatio);
            }

            return new Size(scaledWidth, scaledHeight);
        } else {
            Log.d("VideoActivityCameraHelp", "viewSize or frameSize is null.");
            return null;
        }
    }
}
