package com.griddynamics.video.conf.camera;

import android.app.Activity;
import android.graphics.SurfaceTexture;
import android.util.Size;
import androidx.annotation.Nullable;


public abstract class CameraHelper {
    /** The listener is called when camera start is complete. */
    public interface OnCameraStartedListener {
        /**
         * Called when camera start is complete and the camera-preview frames can be accessed from the
         * surfaceTexture. The surfaceTexture can be null if it is not prepared by the CameraHelper.
         */
        public void onCameraStarted(@Nullable SurfaceTexture surfaceTexture);
    }

    protected static final String TAG = "CameraHelper";

    /** Represents the direction the camera faces relative to device screen. */
    public static enum CameraFacing {
        FRONT,
        BACK
    };

    protected OnCameraStartedListener onCameraStartedListener;

    protected CameraFacing cameraFacing;

    /**
     * Initializes the camera and sets it up for accessing frames from a custom SurfaceTexture object.
     * The SurfaceTexture object can be null when it is the CameraHelper that prepares a
     * SurfaceTexture object for grabbing frames.
     */
    public abstract void startCamera(
            Activity context, CameraFacing cameraFacing, @Nullable SurfaceTexture surfaceTexture);

    /**
     * Computes the ideal size of the camera-preview display (the area that the camera-preview frames
     * get rendered onto, potentially with scaling and rotation) based on the size of the view
     * containing the display. Returns the computed display size.
     */
    public abstract Size computeDisplaySizeFromViewSize(Size viewSize);

    /** Returns a boolean which is true if the camera is in Portrait mode, false in Landscape mode. */
    public abstract boolean isCameraRotated();

    public void setOnCameraStartedListener(@Nullable OnCameraStartedListener listener) {
        onCameraStartedListener = listener;
    }
}
