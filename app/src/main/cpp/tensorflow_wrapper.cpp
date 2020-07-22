/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include <cstring>
#include <jni.h>
#include <cinttypes>
#include <android/log.h>
#include <string>
#include <c_api.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <android/bitmap.h>

TfLiteModel *model;
TfLiteInterpreterOptions *options;
TfLiteInterpreter *interpreter;

#define ALPHA(rgb) (uint8_t)(rgb >> 24)
#define RED(rgb)   (uint8_t)(rgb >> 16)
#define GREEN(rgb) (uint8_t)(rgb >> 8)
#define BLUE(rgb)  (uint8_t)(rgb)

#define UNMULTIPLY(color, alpha) ((0xFF * color) / alpha)
#define BLEND(back, front, alpha) ((front * alpha) + (back * (255 - alpha))) / 255
#define ARGB(a, r, g, b) (a << 24) | ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (b & 0xFF)

const jfloat f255 = 255;
const jint i256 = 256;
#define LOG_TAG "DEMO"

extern "C"
JNIEXPORT jlong
JNICALL
Java_com_griddynamics_video_conf_MainActivity_jniInitModel(
        JNIEnv *env,
        jobject /* this */,
        jstring _assetName,
        jobject _assetManager
) {
    AAssetManager *assetManager = AAssetManager_fromJava(env, _assetManager);
    const char *assetName = env->GetStringUTFChars(_assetName, JNI_FALSE);

    AAsset *asset =
            AAssetManager_open(assetManager, assetName, AASSET_MODE_BUFFER);

    if (asset == nullptr) {
        return 11;
    }
    off_t start;
    off_t length;
    const int fd = AAsset_openFileDescriptor(asset, &start, &length);


    off_t dataSize = AAsset_getLength(asset);
    const void *const memory = AAsset_getBuffer(asset);

    // Use as const char*
    const char *const memChar = (const char *) memory;

    // Create a new Buffer for the FlatBuffer with the size needed.
    // It has to exist alongside the FlatBuffer model as long as the model shall exist!
    // char* flatBuffersBuffer; (declared in the header file of the class in which I use this).
    auto flatBuffersBuffer = new char[dataSize];

    for (int i = 0; i < dataSize; i++) {
        flatBuffersBuffer[i] = memChar[i];
    }

    model = TfLiteModelCreate(flatBuffersBuffer, dataSize);
    options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreterOptionsSetNumThreads(options, 2);

    interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    return model == nullptr ? 555 : 5005;
}

const jsize sizeInputInFloat = 256 * 256 * 3;
const jsize sizeInputInBytes = sizeInputInFloat * sizeof(jfloat);
const jsize sizeOutputInFloat = 256 * 256;
const jsize sizeOutputInBytes = sizeOutputInFloat * sizeof(jfloat);

extern "C"
JNIEXPORT void
JNICALL
Java_com_griddynamics_video_conf_MainActivity_startCompute(
        JNIEnv *env,
        jobject /* this */,
        jlong _nnModel,
        jobject input,
        jobject output
) {
    auto *inputBuf = (jbyte *) env->GetDirectBufferAddress(input);
    auto *outputBuf = (jbyte *) env->GetDirectBufferAddress(output);

    TfLiteTensor *inputTensor = TfLiteInterpreterGetInputTensor(interpreter, 0);

    TfLiteTensorCopyFromBuffer(inputTensor, inputBuf, sizeInputInBytes);

    // Execute inference.
    TfLiteInterpreterInvoke(interpreter);

    // Extract the output tensor data.
    const TfLiteTensor *outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);

    TfLiteTensorCopyToBuffer(outputTensor, outputBuf, sizeOutputInBytes);
}

class JniBitmap {
public:
    uint32_t *_storedBitmapPixels;
    AndroidBitmapInfo _bitmapInfo;

    JniBitmap() {
        _storedBitmapPixels = NULL;
    }
};

extern "C"
JNIEXPORT void
JNICALL
Java_com_griddynamics_video_conf_MainActivity_jniFreeBitmapData(
        JNIEnv *env, jobject obj, jobject handle) {
    auto *jniBitmap = (JniBitmap *) env->GetDirectBufferAddress(handle);
    if (jniBitmap == nullptr || jniBitmap->_storedBitmapPixels == nullptr)
        return;
    delete[] jniBitmap->_storedBitmapPixels;
    jniBitmap->_storedBitmapPixels = nullptr;
    delete jniBitmap;
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_griddynamics_video_conf_MainActivity_jniStoreBitmapData(
        JNIEnv *env, jobject obj, jobject bitmap) {
    AndroidBitmapInfo bitmapInfo;
    uint32_t *storedBitmapPixels = NULL;
    //LOGD("reading bitmap info...");
    int ret;
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &bitmapInfo)) < 0) {
        return NULL;
    }
    //LOGD("width:%d height:%d stride:%d", bitmapInfo.width, bitmapInfo.height, bitmapInfo.stride);
    if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        return NULL;
    }
    //
    //read pixels of bitmap into native memory :
    //
    //LOGD("reading bitmap pixels...");
    void *bitmapPixels;
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, &bitmapPixels)) < 0) {
        return NULL;
    }
    uint32_t *src = (uint32_t *) bitmapPixels;
    storedBitmapPixels = new uint32_t[bitmapInfo.height * bitmapInfo.width];
    int pixelsCount = bitmapInfo.height * bitmapInfo.width;
    memcpy(storedBitmapPixels, src, sizeof(uint32_t) * pixelsCount);
    AndroidBitmap_unlockPixels(env, bitmap);
    JniBitmap *jniBitmap = new JniBitmap();
    jniBitmap->_bitmapInfo = bitmapInfo;
    jniBitmap->_storedBitmapPixels = storedBitmapPixels;
    return env->NewDirectByteBuffer(jniBitmap, 0);
}

extern "C" JNIEXPORT jobject JNICALL
Java_com_griddynamics_video_conf_MainActivity_jniGetBitmapFromStoredBitmapData(
        JNIEnv *env, jobject obj, jobject handle) {
    JniBitmap *jniBitmap = (JniBitmap *) env->GetDirectBufferAddress(handle);
    if (jniBitmap == NULL || jniBitmap->_storedBitmapPixels == NULL) {
        return NULL;
    }
    //
    //creating a new bitmap to put the pixels into it - using Bitmap Bitmap.createBitmap (int width, int height, Bitmap.Config config) :
    //
    jclass bitmapCls = env->FindClass("android/graphics/Bitmap");
    jmethodID createBitmapFunction = env->GetStaticMethodID(bitmapCls,
                                                            "createBitmap",
                                                            "(IILandroid/graphics/Bitmap$Config;)Landroid/graphics/Bitmap;");
    jstring configName = env->NewStringUTF("ARGB_8888");
    jclass bitmapConfigClass = env->FindClass("android/graphics/Bitmap$Config");
    jmethodID valueOfBitmapConfigFunction = env->GetStaticMethodID(
            bitmapConfigClass, "valueOf",
            "(Ljava/lang/String;)Landroid/graphics/Bitmap$Config;");
    jobject bitmapConfig = env->CallStaticObjectMethod(bitmapConfigClass,
                                                       valueOfBitmapConfigFunction, configName);
    jobject newBitmap = env->CallStaticObjectMethod(bitmapCls,
                                                    createBitmapFunction,
                                                    jniBitmap->_bitmapInfo.width,
                                                    jniBitmap->_bitmapInfo.height, bitmapConfig);
    //
    // putting the pixels into the new bitmap:
    //
    int ret;
    void *bitmapPixels;
    if ((ret = AndroidBitmap_lockPixels(env, newBitmap, &bitmapPixels)) < 0) {
        return NULL;
    }
    uint32_t *newBitmapPixels = (uint32_t *) bitmapPixels;
    int pixelsCount = jniBitmap->_bitmapInfo.height
                      * jniBitmap->_bitmapInfo.width;
    memcpy(newBitmapPixels, jniBitmap->_storedBitmapPixels,
           sizeof(uint32_t) * pixelsCount);
    AndroidBitmap_unlockPixels(env, newBitmap);
    //LOGD("returning the new bitmap");
    return newBitmap;
}

int64_t getTimeNsec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t) now.tv_sec * 1000000000LL + now.tv_nsec;
}

extern "C"
JNIEXPORT jobject
JNICALL
Java_com_griddynamics_video_conf_MainActivity_jniProcessBitmap(
        JNIEnv *env, jobject obj,
        jobject output, jobject result,
        jobject bitmap, jobject bitmapBack,
        jint nnewWidth, jint nnewHeight) {

    auto timestamp = getTimeNsec();

    jobject handle = Java_com_griddynamics_video_conf_MainActivity_jniStoreBitmapData(env, obj,
                                                                                      bitmap);
    auto *jniBitmap = (JniBitmap *) env->GetDirectBufferAddress(handle);

    auto *normalized = (jfloat *) env->GetDirectBufferAddress(output);
    auto *outputBuf = (jfloat *) env->GetDirectBufferAddress(result);

    const jint newWidth = i256;
    const jint newHeight = i256;

    auto *newBitmapPixels = new uint32_t[nnewWidth * nnewHeight];

    uint32_t oldWidth = jniBitmap->_bitmapInfo.width;
    uint32_t oldHeight = jniBitmap->_bitmapInfo.height;
    uint32_t *previousData = jniBitmap->_storedBitmapPixels;
    int x2, y2;
    int whereToPut = 0;
    for (int y = 0; y < newHeight; ++y) {
        for (int x = 0; x < newWidth; ++x) {
            x2 = x * oldWidth / newWidth;
            if (x2 < 0)
                x2 = 0;
            else if (x2 >= oldWidth)
                x2 = oldWidth - 1;
            y2 = y * oldHeight / newHeight;
            if (y2 < 0)
                y2 = 0;
            else if (y2 >= oldHeight)
                y2 = oldHeight - 1;

            auto pixel = previousData[(y2 * oldWidth) + x2];

            normalized[whereToPut * 3] = ((pixel & 0xff0000) >> 16) / f255;
            normalized[whereToPut * 3 + 1] = ((pixel & 0x00ff00) >> 8) / f255;
            normalized[whereToPut * 3 + 2] = ((pixel & 0x0000ff)) / f255;
            whereToPut++;
        }
    }

    Java_com_griddynamics_video_conf_MainActivity_jniFreeBitmapData(env, obj, handle);
    auto diff = (getTimeNsec() - timestamp) / 1000000000.0;
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s",
                        ("norm time - " + std::to_string(diff)).c_str());

    timestamp = getTimeNsec();

    Java_com_griddynamics_video_conf_MainActivity_startCompute(env, obj, 0L, output, result);

    diff = (getTimeNsec() - timestamp) / 1000000000.0;
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s",
                        ("compute time - " + std::to_string(diff)).c_str());

    timestamp = getTimeNsec();
    handle = Java_com_griddynamics_video_conf_MainActivity_jniStoreBitmapData(env, obj, bitmapBack);
    jniBitmap = (JniBitmap *) env->GetDirectBufferAddress(handle);

    oldWidth = i256;
    oldHeight = i256;
    previousData = jniBitmap->_storedBitmapPixels;
    whereToPut = 0;
    for (int y = 0; y < nnewHeight; ++y) {
        for (int x = 0; x < nnewWidth; ++x) {
            x2 = x * oldWidth / nnewWidth;
            if (x2 < 0)
                x2 = 0;
            else if (x2 >= oldWidth)
                x2 = oldWidth - 1;
            y2 = y * oldHeight / nnewHeight;
            if (y2 < 0)
                y2 = 0;
            else if (y2 >= oldHeight)
                y2 = oldHeight - 1;

            const uint32_t pixel = previousData[whereToPut];
            const uint32_t alpha = ((1 - outputBuf[(y2 * oldWidth) + x2]) * f255);
            const uint32_t pixelNew = ARGB(
                                              alpha,
                                              0x00000000,
                                              0x00000000,
                                              0x00000000
                                      );
            newBitmapPixels[whereToPut++] = pixelNew;
        }
    }

    delete[] previousData;
    jniBitmap->_storedBitmapPixels = newBitmapPixels;
    jniBitmap->_bitmapInfo.width = nnewWidth;
    jniBitmap->_bitmapInfo.height = nnewHeight;

    diff = (getTimeNsec() - timestamp) / 1000000000.0;
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "%s",
                        ("finish scaling time - " + std::to_string(diff)).c_str());


    return Java_com_griddynamics_video_conf_MainActivity_jniGetBitmapFromStoredBitmapData(env, obj,
                                                                                          handle);
}

extern "C"
JNIEXPORT void
JNICALL
Java_com_griddynamics_video_conf_MainActivity_jniDestroyModel(
        JNIEnv *env,
        jobject /* this */) {
    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);
}