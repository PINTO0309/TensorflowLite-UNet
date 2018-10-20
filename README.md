# Tensorflow Lite-UNet
Implementation of UNet by Tensorflow Lite.  
Semantic segmentation without using GPU with RaspberryPi.  
In order to maximize the learning efficiency of the model, this learns only the "Person" class of VOC2012.  
  
**As of October 18, 2018, README is under construction.**  
  
**=============================================================================  
[October 18, 2018]  
Added correspondence and one class segmentation of TensorflowLite to tks10/segmentation_unet.  
https://github.com/tks10/segmentation_unet.git  
https://qiita.com/tktktks10/items/0f551aea27d2f62ef708  
I confirmed the operation with Tensorflow v1.11.0 and Tensorflow-gpu v1.11.0, Tensorflow Lite v1.11.0.   =============================================================================**  

**【Japanese Article1】 https://qiita.com/PINTO/items/8221d8ccac55baf1f24e**  

## Summary

## Change history
<details><summary>Change history</summary><div>
  
[Oct 19, 2018]　Initial release.<br>

</div></details><br><br>

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
