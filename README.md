# Tensorflow Lite-UNet
**As of October 18, 2018, README is under construction.**  
  
Implementation of UNet by Tensorflow Lite.  
Semantic segmentation without using GPU with RaspberryPi.  
In order to maximize the learning efficiency of the model, this learns only the "Person" class of VOC2012.  
I confirmed the operation with Tensorflow v1.11.0 and Tensorflow-gpu v1.11.0, Tensorflow Lite v1.11.0, Tensorflow Lite-gpu v1.11.0.  
  
I added correspondence and one class segmentation of TensorflowLite to tks10/segmentation_unet.  
**https://github.com/tks10/segmentation_unet.git**  
**https://qiita.com/tktktks10/items/0f551aea27d2f62ef708**  
  
My Japanese Article is below.  
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
### 1. Development environment
- Ubuntu 16.04
- Tensorflow-GPU v1.11.0 <- Performing a test while switching mutually with CPU version, pip command / self build
- Tensorflow-CPU v1.11.0 <- Performing a test while mutually switching with the GPU version, pip command / self build
- Tesla K80 or Geforce GTX 1070 or Quadro P2000
- CUDA 9.0
- cuDNN 7.0
- Python 2.7 or 3.5

### 2. Execution environment
- RaspberryPi3 + Raspbian Stretch
- Bazel 0.17.2
- Tensorflow v1.11.0 <- Self build
- Tensorflow Lite v1.11.0 <- Self build
- Python 2.7 or 3.5
- OpenCV 3.4.2
- MicroSD Card 32GB

## Environment construction procedure
### 1. Construction of learning environment
#### (1) Learning environment introduction to GPU equipped PC Ubuntu 16.04
[1] Execute below. Introduction of CUDA 9.0, cu DNN 7.0.
```
$ cd ~
$ sudo apt-get remove cuda-*
$ sudo apt-get purge cuda-*

# 1.Download cuda-repo-ubuntu1604_9.0.176-1_amd64.deb from NVIDIA
# 2.Download libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb from NVIDIA
# 3.Download libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb from NVIDIA

$ sudo dpkg -i libcudnn7*
$ sudo dpkg -i cuda-*
$ sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
$ sudo apt update
$ sudo apt install cuda-9.0
$ echo 'export PATH=/usr/local/cuda-9.0/bin:${PATH}' >> ~/.bashrc
$ echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
$ source ~/.bashrc
$ sudo ldconfig
$ nvcc -V
$ cd ~;nano cudnn_version.cpp

############################### Paste below. ###################################
#include <cudnn.h>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "CUDNN_VERSION: " << CUDNN_VERSION << std::endl;
    return 0;
}
############################### Paste above ###################################

$ nvcc cudnn_version.cpp -o cudnn_version
$ ./cudnn_version
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 396.44                 Driver Version: 396.44                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 107...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   54C    P0    32W /  N/A |    254MiB /  8119MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1461      G   /usr/lib/xorg/Xorg                           184MiB |
|    0      3364      G   ...quest-channel-token=4480583668747587845    67MiB |
+-----------------------------------------------------------------------------+
```
[2] Introduce Tensorflow with the pip command and Clone the Github repository for learning.
```
$ cd ~
$ sudo pip2 install tensorflow-gpu==1.11.0
$ sudo pip3 install tensorflow-gpu==1.11.0
$ git clone -b pinto0309work https://github.com/PINTO0309/TensorFlow-ENet.git
$ cd TensorFlow-ENet
$ git checkout pinto0309work
```

### 2. Learning ENet and streamlining .pb
#### (1) Learning ENet
Execute the following command only when you want to learn with your own data set.  
If you do not need to learn with your own data set, you can skip this phase.  
Features and learning logic of this model are not touched on this occasion, but please refer Clone repository if you are interested. **[PINTO0309 - Tensorflow-ENet - Github](https://github.com/PINTO0309/TensorFlow-ENet.git)**  
To learn with your own data set, you just deploy your favorite image dataset to a given path before running **`./train.sh`**.  
Many input parameters such as the destination path of the data set are defined in **`train_enet.py`**, so you can process **`train.sh`** according to your preference.
```
$ cd ~/TensorFlow-ENet
$ chmod 777 train.sh
$ ./train.sh
```

#### (2) Slimming the checkpoint file

#### (3) Generate compressed .pb file
### 3. Learning UNet and streamlining .pb
#### (1) Learning UNet
#### (2) UNet learning results
#### (3) Slimming the checkpoint file
#### (4) Generate compressed .pb file
### 4. Execution environment construction
### 5. Operation verification

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
