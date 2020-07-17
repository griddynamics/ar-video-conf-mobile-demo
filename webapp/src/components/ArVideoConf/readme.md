# ArVideoConf

[ArVideoConf](./ArVideoConf.js) is the main implementation of the background segmentation video segmentation and background processing. Other implementations are used for  experimenting only.

#### Model performace

Due to the slow model prediction (~160ms), mask is updated on every 5th frame. It had been tried to improve performance with a web worker, but it only increased the delay (consider trying again).

#### Display performace

[`tf.browser.toPixels`](https://js.tensorflow.org/api/2.0.1/#browser.toPixels) was used for displaying tensors first, but it was very slow (max 20fps for only displaying input tensor without any additional operation).

Someone wrote on github that for the images with smaller resolution rendering time can be decreased if we use `canvasContex.putImageData` with image data as `Uint8ClampedArray`. This is implemented in the `TensorDisplay` class ([util.js](../../util.js)). With this trick rendering performace is significantly improved.

# `ArVideoConfContours`

In [opencv.html](../../../public/opencv.html) it's showed how we can combine [background subtraction](https://docs.opencv.org/3.4/de/df4/tutorial_js_bg_subtraction.html), [edge detection](https://docs.opencv.org/3.4/d7/de1/tutorial_js_canny.html) and [contures](https://docs.opencv.org/3.4/d0/d43/tutorial_js_table_of_contents_contours.html) in order to detect polygons which represent the changes on the image.

#### Posible usage

In [ArVideoConfContours](./ArVideoConfContours.js) it's tried to use detected polygons to update mask instead of frequent model prediction. It's not optimized and have some issues but it might be the way for further performance improvements.
