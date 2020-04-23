package com.griddynamics.video.conf;

import android.graphics.Bitmap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

import static android.content.ContentValues.TAG;

public class Utl {
    public static Bitmap onImageAvailable(ImageReader reader) {
        Log.i(TAG, "in OnImageAvailable");
        FileOutputStream fos = null;
        Bitmap bitmap = null;
        Image img = null;
        try {
            img = reader.acquireLatestImage();
            if (img != null) {
                Image.Plane[] planes = img.getPlanes();
                if (planes[0].getBuffer() == null) {
                    return null;
                }
                int width = img.getWidth();
                int height = img.getHeight();
                int pixelStride = planes[0].getPixelStride();
                int rowStride = planes[0].getRowStride();
                int rowPadding = rowStride - pixelStride * width;
                byte[] newData = new byte[width * height * 4];

                int offset = 0;
                bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                ByteBuffer buffer = planes[0].getBuffer();
                for (int i = 0; i < height; ++i) {
                    for (int j = 0; j < width; ++j) {
                        int pixel = 0;
                        pixel |= (buffer.get(offset) & 0xff) << 16;     // R
                        pixel |= (buffer.get(offset + 1) & 0xff) << 8;  // G
                        pixel |= (buffer.get(offset + 2) & 0xff);       // B
                        pixel |= (buffer.get(offset + 3) & 0xff) << 24; // A
                        bitmap.setPixel(j, i, pixel);
                        offset += pixelStride;
                    }
                    offset += rowPadding;
                }
                img.close();
                return bitmap;
            }

        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (null != fos) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (null != bitmap) {
                bitmap.recycle();
            }
            if (null != img) {
                img.close();
            }

        }
        return null;
    }

}
