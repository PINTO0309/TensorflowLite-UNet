# Tensorflow Lite-UNet
**As of October 18, 2018, README is under construction.**  
  
Implementation of UNet by Tensorflow Lite.  
Semantic segmentation without using GPU with RaspberryPi.  
In order to maximize the learning efficiency of the model, this learns only the "Person" class of VOC2012.  
I confirmed the operation with Tensorflow v1.11.0 and Tensorflow-gpu v1.11.0, Tensorflow Lite v1.11.0, Tensorflow Lite-gpu v1.11.0.  
I also compared with Tensorflow + ENet.  
  
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
[1] Execute below. Introduction of CUDA 9.0, cuDNN 7.0.
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

<details><summary>Shell parameter example·Default value and explanation</summary><div>

```
#Directory arguments
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('logdir', './log/original', 'The log directory to save your checkpoint and event files.')
flags.DEFINE_boolean('save_images', True, 'Whether or not to save your images.')
flags.DEFINE_boolean('combine_dataset', False, 'If True, combines the validation with the train dataset.')

#Training arguments
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 10, 'The batch_size for training.')
flags.DEFINE_integer('eval_batch_size', 25, 'The batch size used for validation.')
flags.DEFINE_integer('image_height', 360, "The input height of the images.")
flags.DEFINE_integer('image_width', 480, "The input width of the images.")
flags.DEFINE_integer('num_epochs', 300, "The number of epochs to train your model.")
flags.DEFINE_integer('num_epochs_before_decay', 100, 'The number of epochs before decaying your learning rate.')
flags.DEFINE_float('weight_decay', 2e-4, "The weight decay for ENet convolution layers.")
flags.DEFINE_float('learning_rate_decay_factor', 1e-1, 'The learning rate decay factor.')
flags.DEFINE_float('initial_learning_rate', 5e-4, 'The initial learning rate for your training.')
flags.DEFINE_string('weighting', "MFB", 'Choice of Median Frequency Balancing or the custom ENet class weights.')

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')
```

</div></details><br>

```
$ cd ~/TensorFlow-ENet
$ chmod 777 train.sh
$ ./train.sh
```

#### (2) Slimming the checkpoint file
It is carried out after learning is over.  
Eliminate all unnecessary variables and optimization processes that are unnecessary at the inference stage, and compress the file size.  
After processing, it is compressed to about one-third of the original file size.  
Since the number part immediately following **`.ckpt-`** is the iteration number of learning, there is a possibility that it differs for each progress of learning.  
Also processed already processed on Clone original repository.  
  
Based on the following three files under the checkpoint folder,  
**`model.ckpt-13800.data-00000-of-00001`**  
**`model.ckpt-13800.index`**  
**`model.ckpt-13800.meta`**  
  
Generate the following four compressed files.  
**`modelfinal.ckpt-13800.data-00000-of-00001`**  
**`modelfinal.ckpt-13800.index`**  
**`modelfinal.ckpt-13800.meta`**  
**`semanticsegmentation_enet.pbtxt`**  
  
Execute the following command.
```
$ python slim_infer.py
```
  
Although I have not done anything special, I just delete almost all the logic at learning and restore -> save.  
For reference, paste the logic below.  
When you learn with your own data set, the ckpt file name is different,  
**`saver.restore(..)`**  
**`saver.save(..)`**  
It is necessary to change the prefix of the ckpt file described in the section before execution.  
In that case, it is unnecessary to specify the part after the number, **`.data-00000-of-00001`** **`.index`** **`.meta`**  
  
<details><summary>The logic of slim_infer.py</summary><div>

```
import tensorflow as tf
from enet import ENet, ENet_arg_scope
slim = tf.contrib.slim

def main():

    graph = tf.Graph()
    with graph.as_default():

        with slim.arg_scope(ENet_arg_scope()):
            inputs = tf.placeholder(tf.float32, [None, 360, 480, 3], name="input") 
            logits, probabilities = ENet(inputs,
                                         12,
                                         batch_size=1,
                                         is_training=False,
                                         reuse=None,
                                         num_initial_blocks=1,
                                         stage_two_repeat=2,
                                         skip_connections=False)

        saver = tf.train.Saver(tf.global_variables())
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, './checkpoint/model.ckpt-13800')
        saver.save(sess, './checkpoint/modelfinal.ckpt-13800')

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, './checkpoint', 'semanticsegmentation_enet.pbtxt', as_text=True)

if __name__ == '__main__':
    main()
```

</div></details><br>

#### (3) Generate compressed .pb file
Execute the following command to generate compressed .pb file from .pbtxt.  
Since two types of Output Nodes are described in comma-separated form, the inference result can receive output from two kinds of Nodes.  
When executing the command, 1.78 MB **`semanticsegmentation_enet.pb`** file is generated under the **`checkpoint`** folder.  
```
$ python freeze_graph.py \
--input_graph=checkpoint/semanticsegmentation_enet.pbtxt \
--input_checkpoint=checkpoint/modelfinal.ckpt-13800 \
--output_graph=checkpoint/semanticsegmentation_enet.pb \
--output_node_names=ENet/fullconv/BiasAdd,ENet/logits_to_softmax \
--input_binary=False
```

### 3. Learning UNet and streamlining .pb
#### (1) Learning UNet
Execute the following command to Clone the learning repository.  
All images are extracted only for the image files that show "Person", and they are already deployed in the dataset folder.  
[PINTO0309 - TensorflowLite-UNet - Github](https://github.com/PINTO0309/TensorflowLite-UNet/tree/master/data_set/VOCdevkit/person)  
  
The method to extract only the "Person" image is to download the VOC2012 dataset and extract it under the folder generated after decompressing,
It is sufficient to extract all the images with the flag "1" set in **`VOCdevkit/VOC2012/ImageSets/Main/person_train.txt`** or **`VOCdevkit/VOC2012/ImageSets/Main/person_trainval.txt`**.  
Even when extracting images other than "Person", it is possible to extract from the file name and extract with the same number.  
Note that it is necessary to extract files one by one from each folder while synchronizing files of the same name from both the **`JPEGImages`** folder and **`SegmentationClass`** folder.  
```
$ cd ~
$ git clone https://github.com/PINTO0309/TensorflowLite-UNet.git
```
With only the training image of VOC2012, the number of samples is as small as 794, which is a major over-learning.  
Execute the following program and inflate the image file by 20 times.  
The inflating operation is carried out at random as follows.  
- Smoothing  
- Gaussian noise addition  
- Salt & Pepper noise addition  
- Rotation  
- Inversion  
  
The image after padding is saved in **`data_set/VOCdevkit/person/JPEGImagesOUT`** and **`data_set/VOCdevkit/person/SegmentationClassOUT`**.  
You can adjust multiples of padding by changing the number of **`increase_num = 20`** to the number you like.  
However, since the number of images that can be processed at one time depends on the VRAM capacity of the GPU, it is necessary to adjust the numerical value small if OutOfMemory appears.  
It is necessary to check the line of limitation that excessive learning does not occur as much as possible and OutOfMemory does not occur.  
```
$ cd TensorflowLite-UNet
$ python3 IncreaseImage.py
```
  
Start learning with the following command.
```
$ python3 main.py --gpu --augmentation --batchsize 32 --epoch 50
```
When reducing the model size, it can be generated by reducing the number of filters in the Convolution layer of **`model.py`** and **`model_infer.py`** to 1/4 as shown below.  

<details><summary>Changes to model.py when reducing model size</summary><div>

```
        conv1_1 = UNet.conv(inputs, filters=8, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv1_2 = UNet.conv(conv1_1, filters=8, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool1 = UNet.pool(conv1_2)

        conv2_1 = UNet.conv(pool1, filters=16, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv2_2 = UNet.conv(conv2_1, filters=16, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool2 = UNet.pool(conv2_2)

        conv3_1 = UNet.conv(pool2, filters=32, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv3_2 = UNet.conv(conv3_1, filters=32, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool3 = UNet.pool(conv3_2)

        conv4_1 = UNet.conv(pool3, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv4_2 = UNet.conv(conv4_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool4 = UNet.pool(conv4_2)

        conv5_1 = UNet.conv(pool4, filters=256, l2_reg_scale=l2_reg)
        conv5_2 = UNet.conv(conv5_1, filters=256, l2_reg_scale=l2_reg)
        concated1 = tf.concat([UNet.conv_transpose(conv5_2, filters=128, l2_reg_scale=l2_reg), conv4_2], axis=3)

        conv_up1_1 = UNet.conv(concated1, filters=128, l2_reg_scale=l2_reg)
        conv_up1_2 = UNet.conv(conv_up1_1, filters=128, l2_reg_scale=l2_reg)
        concated2 = tf.concat([UNet.conv_transpose(conv_up1_2, filters=64, l2_reg_scale=l2_reg), conv3_2], axis=3)

        conv_up2_1 = UNet.conv(concated2, filters=64, l2_reg_scale=l2_reg)
        conv_up2_2 = UNet.conv(conv_up2_1, filters=64, l2_reg_scale=l2_reg)
        concated3 = tf.concat([UNet.conv_transpose(conv_up2_2, filters=32, l2_reg_scale=l2_reg), conv2_2], axis=3)

        conv_up3_1 = UNet.conv(concated3, filters=32, l2_reg_scale=l2_reg)
        conv_up3_2 = UNet.conv(conv_up3_1, filters=32, l2_reg_scale=l2_reg)
        concated4 = tf.concat([UNet.conv_transpose(conv_up3_2, filters=16, l2_reg_scale=l2_reg), conv1_2], axis=3)

        conv_up4_1 = UNet.conv(concated4, filters=16, l2_reg_scale=l2_reg)
        conv_up4_2 = UNet.conv(conv_up4_1, filters=16, l2_reg_scale=l2_reg)
        outputs = UNet.conv(conv_up4_2, filters=ld.DataSet.length_category(), kernel_size=[1, 1], activation=None, name="output")
```

</div></details><br>
<details><summary>Changes to model_infer.py when reducing model size</summary><div>

```
        conv1_1 = UNet.conv(inputs, filters=8, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        conv1_2 = UNet.conv(conv1_1, filters=8, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        pool1 = UNet.pool(conv1_2)

        # 1/2, 1/2, 64
        conv2_1 = UNet.conv(pool1, filters=16, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        conv2_2 = UNet.conv(conv2_1, filters=16, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        pool2 = UNet.pool(conv2_2)

        # 1/4, 1/4, 128
        conv3_1 = UNet.conv(pool2, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        conv3_2 = UNet.conv(conv3_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        pool3 = UNet.pool(conv3_2)

        # 1/8, 1/8, 256
        conv4_1 = UNet.conv(pool3, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        conv4_2 = UNet.conv(conv4_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=False)
        pool4 = UNet.pool(conv4_2)

        # 1/16, 1/16, 512
        conv5_1 = UNet.conv(pool4, filters=256, l2_reg_scale=l2_reg)
        conv5_2 = UNet.conv(conv5_1, filters=256, l2_reg_scale=l2_reg)
        concated1 = tf.concat([UNet.conv_transpose(conv5_2, filters=128, l2_reg_scale=l2_reg), conv4_2], axis=3)

        conv_up1_1 = UNet.conv(concated1, filters=128, l2_reg_scale=l2_reg)
        conv_up1_2 = UNet.conv(conv_up1_1, filters=128, l2_reg_scale=l2_reg)
        concated2 = tf.concat([UNet.conv_transpose(conv_up1_2, filters=64, l2_reg_scale=l2_reg), conv3_2], axis=3)

        conv_up2_1 = UNet.conv(concated2, filters=64, l2_reg_scale=l2_reg)
        conv_up2_2 = UNet.conv(conv_up2_1, filters=64, l2_reg_scale=l2_reg)
        concated3 = tf.concat([UNet.conv_transpose(conv_up2_2, filters=32, l2_reg_scale=l2_reg), conv2_2], axis=3)

        conv_up3_1 = UNet.conv(concated3, filters=32, l2_reg_scale=l2_reg)
        conv_up3_2 = UNet.conv(conv_up3_1, filters=32, l2_reg_scale=l2_reg)
        concated4 = tf.concat([UNet.conv_transpose(conv_up3_2, filters=16, l2_reg_scale=l2_reg), conv1_2], axis=3)

        conv_up4_1 = UNet.conv(concated4, filters=16, l2_reg_scale=l2_reg)
        conv_up4_2 = UNet.conv(conv_up4_1, filters=16, l2_reg_scale=l2_reg)
        outputs = UNet.conv(conv_up4_2, filters=ld.DataSet.length_category(), kernel_size=[1, 1], activation=None, name="output")
```

</div></details><br>

#### (2) UNet learning results
**Large size model**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/15.png" width="720">  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/16.png" width="720">  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/17.png" width="720">  
  
**Small size model**  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/18.png" width="720">  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/19.png" width="720">  
<img src="https://github.com/PINTO0309/TensorflowLite-UNet/raw/master/media/20.png" width="720">  


#### (3) Slimming the checkpoint file
Execute the following command and compress the file size by excluding all unnecessary variables and optimization processing which become unnecessary at the reasoning stage as in ENet.  
The checkpoint file is generated under the **`model`** folder.  
```
$ python3 main_infer.py
```

<details><summary>The logic of main_infer.py</summary><div>

```
import tensorflow as tf
from util import model_infer as model

#######################################################################################
### $ python3 main_infer.py
#######################################################################################

def main():

    graph = tf.Graph()
    with graph.as_default():

        model_unet = model.UNet(l2_reg=0.0001).model

        saver = tf.train.Saver(tf.global_variables())
        sess  = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        print("in=", model_unet.inputs.name)
        print("on=", model_unet.outputs.name)

        saver.restore(sess, './model/deploy.ckpt')
        saver.save(sess, './model/deployfinal.ckpt')

        graphdef = graph.as_graph_def()
        tf.train.write_graph(graphdef, './model', 'semanticsegmentation_person.pbtxt', as_text=True)

if __name__ == '__main__':
    main()
```

</div></details><br>

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
