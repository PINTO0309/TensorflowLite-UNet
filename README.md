# Tensorflow Lite-UNet
Implementation of UNet by Tensorflow Lite.  
Semantic segmentation without using GPU with RaspberryPi.  
In order to maximize the learning efficiency of the model, this learns only the "Person" class of VOC2012.  
I confirmed the operation with Tensorflow v1.11.0 and Tensorflow-gpu v1.11.0, Tensorflow Lite v1.11.0.  
  
**As of October 18, 2018, README is under construction.**  
  
**=============================================================================  
[October 18, 2018]  
Added correspondence and one class segmentation of TensorflowLite to tks10/segmentation_unet.  
https://github.com/tks10/segmentation_unet.git  
https://qiita.com/tktktks10/items/0f551aea27d2f62ef708  
=============================================================================**  

**【Japanese Article】 [Tensorflow Lite / Tensorflow + RaspberryPi + Python implements ultra light "Semantic Segmentation" model "UNet" "ENet" Lightweight Model Part 2](https://qiita.com/PINTO/items/8221d8ccac55baf1f24e)**  

## Summary

## Change history
<details><summary>Change history</summary><div>
  
[Oct 19, 2018]　Initial release.<br>

</div></details><br>

## Inference images
**＜Test generated image by GPU version Tensorflow v1.11.0 + UNet + Ubuntu16.04 PC＞  
※ Size of learning model 31.1 MB  
※ Geforce GTX 1070 Inference time 1.83 sec**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/01.jpeg" width="380"> <img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/02.jpeg" width="380">  
  
**＜Test generated image by CPU version Tensorflow Lite v1.11.0 + UNet + Ubuntu16.04 PC＞  
※ Size of learning model 9.9 MB  
※ Inference time in 8th generation Corei7 1.13 sec**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/03.jpeg" width="380"> <img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/04.jpeg" width="380">
  
**＜Test generated image by GPU version Tensorflow Lite v1.11.0 (Self build) + UNet + Ubuntu16.04 PC Part 1＞  
※ Size of learning model 9.9 MB  
※ Inference time with Geforce GTX 1070 0.87 sec  
※ CUDA 9.0 and cuDNN 7.0 are valid**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/05.jpeg" width="380"> <img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/06.jpeg" width="380">
  
**＜Test generated image by GPU version Tensorflow Lite v1.11.0 (Self build) + UNet + Ubuntu16.04 PC Part 2＞  
※ Size of learning model 625 KB  
※ Inference time with Geforce GTX 1070 0.07 sec (70 ms)  
※ CUDA 9.0 and cuDNN 7.0 are valid**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/07.jpeg" width="380"> <img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/08.jpeg" width="380">
  
**＜Test generated image by CPU version Tensorflow v1.11.0 (Self build) + ENet + RaspberryPi3＞  
※ Size of learning model 1.87 MB  
※ Inference time with ARM Cortex-A53 10.2 sec  
※ Using Tensorflow v1.11.0 introduced with the pip command results in "Bus error" and terminates abnormally.**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/09.jpeg" width="380"> <img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/10.jpeg" width="380">
  
**＜Test generated image by CPU version Tensorflow v1.11.0 (Self build) + UNet + RaspberryPi3 Part 1＞  
※ Size of learning model 9.9 MB  
※ Inference time with ARM Cortex-A53 11.76 sec  
※ Despite the model size of 5 times or more of ENet, there is no difference for only 1.5 seconds.  
※ Slow but precise.**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/11.jpeg" width="380"> <img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/12.jpeg" width="380">
  
**＜Test generated image by CPU version Tensorflow v1.11.0 (Self build) + UNet + RaspberryPi3 Part 2＞  
※ Size of learning model 625 KB  
※ Inference time with ARM Cortex-A53 0.47 sec  
※ Fast but less accurate.**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/13.jpeg" width="380"> <img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/14.jpeg" width="380">  
## Environment

## Environment construction procedure

## Reference article, thanks
**https://qiita.com/tktktks10/items/0f551aea27d2f62ef708**  
**https://github.com/tks10/segmentation_unet.git**  
**http://blog.gclue.com/?p=836**  
**https://stackoverflow.com/questions/50902067/how-to-import-the-tensorflow-lite-interpreter-in-python**  
**https://heartbeat.fritz.ai/compiling-a-tensorflow-lite-build-with-custom-operations-cf6330ee30e2**  
**http://tensorflow.classcat.com/category/tensorflow-lite/**  
**http://tensorflow.classcat.com/2016/03/04/tensorflow-how-to-adding-a-new-op/**  
**https://tyfkda.github.io/blog/2016/09/14/tensorflow-protobuf.html**  
**https://stackoverflow.com/questions/52400043/how-to-get-toco-to-work-with-shape-none-24-24-3**  
**https://groups.google.com/a/tensorflow.org/forum/#!msg/tflite/YlTLq9fGnvE/SVfhSbklBAAJ**  
**https://www.tensorflow.org/lite/rpi**  
**https://www.tensorflow.org/api_docs/python/tf/contrib/lite/TocoConverter**  
**https://www.tensorflow.org/lite/convert/python_api**  
**https://www.tensorflow.org/lite/devguide**  
**https://www.tensorflow.org/lite/convert/cmdline_examples**  
**https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite**  
**https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/g3doc/custom_operators.md**  
**https://github.com/tensorflow/tensorflow/issues/21574**  
**https://docs.bazel.build/versions/master/command-line-reference.html**  
