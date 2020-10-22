# ar-video-conf-mobile-demo
IT-63822

# Android part

## Apks:
**cto_rnd_ar_video_conf_v1.apk** 
Branch: main
Git hash: a26d8eead0ee2b07ee2246b9c5ecf0cb6decb65f
Initial state of the project, app is laggy and occasionally fails, code is bad formatted
Features:
* Face detection (Standart deeplabv3_257_mv_gpu.tflite is used)
* Palm detection (Based on mediapipe)
* Combined (Face detection + hand gesture recognition)
* Custom model (Not working)
* Static analyzer (Has to give output of the model, but doesn't work)

**cto_rnd_ar_video_conf_v2.apk** 
Branch: main
Git hash: b0a06138867035f829b1b02d8ff1492dc5a8df2a
Changes:
* Custom model is fixed, but app is still occasionally fails

**cto_rnd_ar_video_conf_v3.apk** 
Branch: main
Git hash: e97f83027027db000d1c293c34dd6be04be9e268
App codebase cleaned and rebuilded and has better video processing in about 100 times. App simplified to single page application with only most wanted features. 
Features:
* Change background with help of our custom model
* Change background with help of DeepLab model
* Add blur with help of our custom model

**cto_rnd_ar_video_conf_v3.apk** 
Branch: main
Git hash: a53df67bb54ab6de5d1adaec47eac0d600edb227
Switched to material design.  
Changes:
* Crashlitic
* By FAB click last model output picture goes to shared bucket for debug purpose
* Option to enable/disable statistic to be sent to Cloud Firestore (includes fps, total processing per image, model inference)
* Option to dynamic threshold change
* Option to dynamic change model scaling
* Option to choose model from list
* Option to revert to default settings

**cto_rnd_ar_video_conf_cpp.apk**
Branch: feature/cpp
Image processing implemented on cpp with few algoritmic optimizations.

Normally(with use of Tensorflow Java Api) we have such steps: 
1. Scaling original image
2. Normalizing
3. Inference
4. Convert output to mask bitmap
5. Scale mask 
6. Apply mask on original image

After optimization, amount of steps and loops was reduced:
1. Scaling and normalizing at the same time
2. Inference
3. Convert output to mask bitmap, scale mask and apply mask on original image 

With lower model density this approach significantly reduces amount of advantage it gives. More details in "charts" folder

**cto_rnd_ar_video_conf_hand_detection_mediapipe.apk**
Branch: feature/hand-mediapipe
Hand detection implemented based on mediapipe

**cto_rnd_ar_video_conf_hand_detection.apk**
Branch: feature/hand-detection
Same sand detection implemented based on mediapipe, but different approach

## Charts:
**Devices comparison** - Total Processing, Inference and FPS for 6 different devices we had (same model)
**Different implementations execution time comparison** - Java API vs Optimized C++ API performance comparision 
**Models execution time comparison** - 10 custom model compared by Total Processing and Inference

# References:
* All files related can be found [here](https://drive.google.com/drive/u/1/folders/1Zw2r5wvcP6VtWz01973lwx3IG-NFFTrA)
* [Article](https://docs.google.com/document/d/1GQnbz9UvCF8TnmRPPuiLMIMqVyBIWslkW_s3GLIFsz0/edit?usp=sharing) about this project 