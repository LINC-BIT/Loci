# Loci
![](https://github.com/LINC-BIT/Loci/blob/main/Method.png)
## Table of contents
- [1 Introduction](#1-introduction)
- [2 How to get started](#2-how-to-get-started)
  * [2.1 Setup](#21-setup)
  * [2.2 Usage](#22-usage)
- [3 Supported models](#3-supported-models-in-image-classification)
- [4 Experiments setting](#4-Experiments-setting)
  * [4.1 Generate task](#41-Generate-task)
  * [4.2 Selection of model](#43-Selection-of-model)
- [5 Experiments](#5-Experiments)
  * [5.1 Running on Cifar100](#51-under-different-workloads-model-and-dataset)
  * [5.2 Running on MiniImageNet](#52-under-different-network-bandwidths)
  * [5.3 Running on TinyImageNet](#53-large-scale)
  * [5.4 Running on ASC](#54-long-task-sequence)
  * [5.5 Running on DSC](#55-under-different-parameter-settings)
  * [5.6 Result](#56-result)
- [6 Citations](#6-citation)

## 1 Introduction
Loci is designed to achieve SOTA performance (accuracy, time, and communication cost etc.) in federated continual learning setting(heterogeneous tasks). It now supports 17 typical neural models under four applications(CV, NLP, Graph and Multimodal). 
For CV application, it supports 5 models:
- [ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html): this model consists of multiple convolutional layers and pooling layers that extract the information in image. Typically, ResNet suffers from gradient vanishing (exploding) and performance degrading when the network is  deep. ResNet thus adds BatchNorm to alleviate gradient vanishing (exploding) and adds residual connection to alleviate the performance degrading.
- [MobileNet](https://arxiv.org/abs/1801.04381): MobileNet is a lightweight convolutional network which widely uses the depthwise separable convolution.
- [DenseNet](https://arxiv.org/pdf/1707.06990.pdf): DenseNet extends ResNet by adding connections between each blocks to aggregate all multi-scale features.
- [WideResNet](): WideResNet (Wide Residual Network) is a deep learning model that builds on the ResNet architecture by increasing the width of residual blocks (using more feature channels) to improve performance and efficiency while reducing the depth of the network.
- [Vit](): The Vision Transformer (ViT) applies the Transformer architecture to image recognition tasks. It segments the image into multiple patches, then inputs these small blocks as sequence data into the Transformer model, using the self-attention mechanism to capture global and local information within the image, thereby achieving efficient image classification.

For NLP application, it cupports 6 models:
- [RNN](): RNN (Recurrent Neural Network) is a type of neural network specifically designed for sequential data, excelling at handling time series and natural language with temporal dependencies.
- [LSTM](): LSTM (Long Short-Term Memory) is a special type of RNN that can learn long-term dependencies, suitable for tasks like time series analysis and language modeling.
- [Bert](): BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained language representation model based on the Transformer architecture, which captures contextual information in text through deep bidirectional training. The BERT model excels in natural language processing (NLP) tasks and can be used for various applications such as text classification, question answering systems, and named entity recognition.
- LSTMMoE():LSTMMoE (LSTM with Mixture of Experts) is a model that combines Long Short-Term Memory (LSTM) networks with a Mixture of Experts framework to enhance sequence modeling by dynamically selecting specialized expert networks for different input patterns.
- GPT2(): GPT-2 (Generative Pre-trained Transformer 2) is a large-scale language model developed by OpenAI, designed to generate coherent and contextually relevant text by leveraging a transformer-based architecture and pre-training on diverse internet data.
- GPTNeo(): GPT-Neo is an open-source language model developed by EleutherAI, designed as an alternative to GPT, utilizing transformer-based architecture to generate high-quality, contextually relevant text.

For Graph application, it supports 2 models:
- [GCN](): GCN (Graph Convolutional Network) is a deep learning model designed to effectively learn representations from graph-structured data by aggregating and transforming information from neighboring nodes through graph convolution operations.
- [GAT](): GAT (Graph Attention Network) is a deep learning model that leverages attention mechanisms to assign learnable weights to neighbor nodes, enabling more effective feature aggregation and representation learning on graph-structured data.

For Multimodel application, it supports 1 model:
- [CLIP](): CLIP (Contrastive Language–Image Pretraining) is a multimodal model developed by OpenAI that learns to associate images and text by jointly training on a large dataset of image-text pairs, enabling powerful zero-shot learning for diverse vision and language tasks.

## 2 How to get started
### 2.1 Setup
**Requirements**
- Edge devices such as Jetson AGX, Jetson TX2, Jetson Xavier NX, Jetson Nano and Rasperry Pi.
- Linux and Windows 
- Python 3.6+
- PyTorch 1.9+
- CUDA 10.2+ 

**Preparing the virtual environment**

1. Create a conda environment and activate it.
	```shell
	conda create -n FedKNOW python=3.7
	conda active FedKNOW
	```
	
2. Install PyTorch 1.9+ in the [offical website](https://pytorch.org/). A NVIDIA graphics card and PyTorch with CUDA are recommended.

  ![image](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/ec360791671f4a4ab322eb4e71cc9e62~tplv-k3u1fbpfcp-zoom-1.image)

3. Clone this repository and install the dependencies.
  ```shell
  git clone https://github.com/LINC-BIT/Loci.git
  pip install -r requirements.txt
  ```
### 2.2 Usage
Run PuzzleFL or the baselines:
```shell
python Loci/main_Loci.py(or other baselines) --dataset [dataset] --model [mdoel]
--num_users [num_users]  --shard_per_user [shard_per_user] --frac [frac] 
--local_bs [local_bs] --lr [lr] --task [task] --epoch [epoch]  --local_ep 
[local_ep] --local_local_ep [local_local_ep] --store_rate [store_rate] 
--select_grad_num [select_grad_num] --gpu [gpu]
```
Arguments:

- `dataset` : the dataset, e.g. `cifar100`, `MiniImageNet`, `TinyImageNet`, `ASC`, `DSC`

- `model`: the model, e.g. `6-Layers CNN`, `ResNet18`, `DenseNet`, `MobiNet`, `RNN`, `LSTM`, `Bert`

- `num_users`: the number of clients

- `shard_per_user`: the number of classes in each client

- `neighbor_nums`: the number of clients per neighbor

- `local_bs`: the batch size in each client

- `lr`: the learning rate

- `task`: the number of tasks

- `epochs`: the number of communications between each client

- `local_ep`:the number of epochs in clients

- `local_local_ep`:the number of updating the local parameters in clients

- `store_rate`: the store rate of model parameters in FedKNOW

- `select_grad_num`: the number of choosing the old grad in FedKNOW

- `gpu`: GPU id

  More details refer to `utils/option.py`.

## 4 Experiments setting
### 4.1 Generate task
#### 4.1.1 Dataset introduction
- [Cifar100](http://www.cs.toronto.edu/~kriz/cifar.html): Cifar100 dataset  has a total of 50000 training samples (500 ones per class) and 10000 test samples (100 ones per class) in 100 different classes.
- [MiniImageNet](https://image-net.org/download.php):MiniImageNet dataset has a total of 50000 training samples (500 ones per class) and 10000 test samples (100 ones per class) in 100 different classes.
- [TinyImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip): TinyImageNet dataset has a total of 100000 training samples (500 ones per class) and 10000 test samples (50 ones per class) in 200 different classes.
- [OnlineShopping](): OnlineShopping data has a total of 100000 training samples (500 ones per class) and 10000 test samples (50 ones per class) in 100 different classes.
- [ASC](http://www.cs.toronto.edu/~kriz/cifar.html): ASC dataset has a total of 95000 training samples (500 ones per class) and 9500 test samples (100 ones per class) in 100 different classes.
- [DSC](https://image-net.org/download.php): DSC dataset has a total of 50000 training samples (500 ones per class) and 10000 test samples (100 ones per class) in 100 different classes.
- [MiniGC](https://image-net.org/download.php): MiniGC dataset has a total of 40000 training samples (500 ones per class) and 8000 test samples (100 ones per class) in 80 different classes.
- [Reddit](https://image-net.org/download.php): Reddit dataset has a total of 40000 training samples (500 ones per class) and 8000 test samples (100 ones per class) in 80 different classes.
- [Cifar100-text](http://www.cs.toronto.edu/~kriz/cifar.html): Cifar100-text dataset  has a total of 50000 training samples (500 ones per class) and 10000 test samples (100 ones per class) in 100 different classes.

#### 4.1.2 Task split method
According to the definition of tasks, we use the continual learning [dataset splitting method](https://openaccess.thecvf.com/content_cvpr_2017/html/Rebuffi_iCaRL_Incremental_Classifier_CVPR_2017_paper.html) to split these datasets into multiple tasks. Each tasks have data samples of different class and is assigned a unique task ID. 
Before building the dataloader, we split each dataset, as follows:
- split Cifar100 into 10 tasks
	```shell
	python dataset/Cifar100.py --task_number=10 --class_number=100
	```
- split MiniImageNet into 10 tasks
	```shell
	python dataset/miniimagenet.py --task_number=10 --class_number=100
	```
- split TinyImageNet into 20 tasks
	```shell
	python dataset/tinyimagenet.py --task_number=20 --class_number=200
	```
- split ASC into 19 tasks
	```shell
	python dataset/asc.py --task_number=19 --class_number=190
	```
- split DSC into 10 tasks
	```shell
	python dataset/dsc.py --task_number=20 --class_number=100
	```
- split OnlineShopping into 10 tasks
	```shell
	python dataset/OnlineShopping.py --task_number=10 --class_number=100
	```
- split MiniGC into 10 tasks
	```shell
	python dataset/MiniGC.py --task_number=10 --class_number=100
	```
- split Reddit into 10 tasks
	```shell
	python dataset/Reddit.py --task_number=10 --class_number=100
	```
#### 4.1.3 Task allocation method
Under the setting of FCL, each client has its own private task sequence, so we allocate each task to all clients in the form of Non-IID according to the method of [FedRep](http://proceedings.mlr.press/v139/collins21a). 
Specifically, we assign the task sequence of each dataset split to all clients. For each task, each client randomly selects 2-5 classes of data, and randomly obtains 10% of the training samples and test samples from the selected classes. As follows:
```shell
def noniid(dataset, num_users, shard_per_user, num_classes, dataname, rand_set_all=[]):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    idxs_dict = {}
    count = 0
    for i in range(len(dataset)):
        if dataname == 'miniimagenet' or dataname == 'FC100' or dataname == 'tinyimagenet':
            label = torch.tensor(dataset.data[i]['label']).item()
        elif dataname == 'Corn50':
            label = torch.tensor(dataset.data['label'][i]).item()
        else:
            label = torch.tensor(dataset.data[i][1]).item()
        if label < num_classes and label not in idxs_dict.keys():
            idxs_dict[label] = []
        if label < num_classes:
            idxs_dict[label].append(i)
            count += 1

    shard_per_class = int(shard_per_user * num_users / num_classes)
    samples_per_user = int( count/num_users )
    # whether to sample more test samples per user
    if (samples_per_user < 20):
        double = True
    else:
        double = False

    for label in idxs_dict.keys():
        x = idxs_dict[label]
        num_leftover = len(x) % shard_per_class
        leftover = x[-num_leftover:] if num_leftover > 0 else []
        x = np.array(x[:-num_leftover]) if num_leftover > 0 else np.array(x)
        x = x.reshape((shard_per_class, -1))
        x = list(x)

        for i, idx in enumerate(leftover):
            x[i] = np.concatenate([x[i], [idx]])
        idxs_dict[label] = x

    if len(rand_set_all) == 0:
        rand_set_all = list(range(num_classes)) * shard_per_class
        random.shuffle(rand_set_all)
        rand_set_all = np.array(rand_set_all).reshape((num_users, -1))

    # divide and assign
    testb = False
    for i in range(num_users):
        if double:
            rand_set_label = list(rand_set_all[i]) * 50
        else:
            rand_set_label = rand_set_all[i]
        rand_set = []
        for label in rand_set_label:
            idx = np.random.choice(len(idxs_dict[label]), replace=False)
            if (samples_per_user < 100 and testb):
                rand_set.append(idxs_dict[label][idx])
            else:
                rand_set.append(idxs_dict[label].pop(idx))
        dict_users[i] = np.concatenate(rand_set)

    test = []
    return dict_users, rand_set_all
```
### 4.2 Selection of model
PuzzleFL supports a variety of models and can easily add new ones. Based on PyTorch, simply specify the number of tasks and the total number of categories in the model.
```shell
class SixCNN(nn.Module):
    def __init__(self, inputsize, outputsize=100,nc_per_task = 10):
        super().__init__()
        self.outputsize = outputsize
        self.feature_net = CifarModel(inputsize)
        self.last = nn.Linear(1024, outputsize, bias=False)
        self.nc_per_task = nc_per_task
    def forward(self, x, t=-1, pre=False, is_cifar=True, avg_act=False):
        h,hidden = self.feature_net(x, avg_act)
        output = self.last(h)
        if is_cifar and t != -1:
            # make sure we predict classes within the current task
            if pre:
                offset1 = 0
                offset2 = int(t * self.nc_per_task)
            else:
                offset1 = int(t * self.nc_per_task)
                offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.outputsize:
                output[:, offset2:self.outputsize].data.fill_(-10e10)
        if avg_act is True:
            return output,hidden
        return output
```

## 5 Experiment
### 5.1 Running on Cifar100
We selected 10 Jetson and raspberry Pi devices with different memory and different computing speeds to test on cifar100, including 2 Jetson-nano devices with 4GB memory, 2 Jetson-Xavier-NX with 16GB memory, 2 Jetson-AgX with 32GB memory and rasberry pi with 4GB memory.
- **Launch the server:**
```shell
## Run on 20 Jetson devices
python multi/server.py --epochs=150 --num_users=20 --frac=0.4 --ip=127.0.0.1:8000
```
**Note：--ip=127.0.0.1:8000 here means that the local machine is used as the center server. If there is an existing server, it can be replaced with the IP address of the server.**

- **Launch the clients:**
* 6-layer CNN on Cifar100
    ```shell
   ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
   do
       python multi/ClientTrain.py --client_id=$i --model=6_layerCNN --dataset=cifar100 --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
   done
   ```
* ResNet18 on Cifar100
   ```shell
     ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
   do
       python multi/ClientTrain.py --client_id=$i --model=ResNet --dataset=cifar100 --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
   done
   ```
**Note:** Please keep the IP addresses of the server and the client consistent. If there are multiple devices running, run the corresponding code directly on the corresponding edge device and replace it with the IP address of the server. The operating instructions of other baselines are in `scripts/difwork`.

### 5.2 Running on MiniImgaeNet
We selected 10 Jetson and rasberry devices with different memory and different computing speeds to test on cifar100, including 2 Jetson-nano devices with 4GB memory, 2 Jetson-Xavier-NX with 16GB memory, 2 Jetson-AgX with 32GB memory and rasberry pi with 4GB memory.
- **Launch the server:**
```shell
## Run on 20 Jetson devices
python multi/server.py --epochs=150 --num_users=20 --frac=0.4 --ip=127.0.0.1:8000
```
**Note：--ip=127.0.0.1:8000 here means that the local machine is used as the center server. If there is an existing server, it can be replaced with the IP address of the server.**

- **Launch the clients:**
* MobiNet on MiniImgaeNet
    ```shell
   ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
   do
       python multi/ClientTrain.py --client_id=$i --model=Mobinet --dataset=MiniImgaeNet --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
   done
   ```
* DenseNet on MiniImgaeNet
   ```shell
     ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
   do
       python multi/ClientTrain.py --client_id=$i --model=DenseNet --dataset=MiniImgaeNet --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
   done
   ```
**Note:** Please keep the IP addresses of the server and the client consistent. If there are multiple devices running, run the corresponding code directly on the corresponding edge device and replace it with the IP address of the server. The operating instructions of other baselines are in `scripts/difwork`.

### 5.3 Running on TinyImageNet
We selected 10 Jetson and rasberry devices with different memory and different computing speeds to test on cifar100, including 2 Jetson-nano devices with 4GB memory, 2 Jetson-Xavier-NX with 16GB memory, 2 Jetson-AgX with 32GB memory and rasberry pi with 4GB memory.
- **Launch the server:**
```shell
## Run on 10 Jetson devices
python multi/server.py --epochs=150 --num_users=10 --frac=0.4 --ip=127.0.0.1:8000
```
**Note：--ip=127.0.0.1:8000 here means that the local machine is used as the center server. If there is an existing server, it can be replaced with the IP address of the server.**

- **Launch the clients:**
* Vit on TinyImageNet
    ```shell
   ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
   do
       python multi/ClientTrain.py --client_id=$i --model=vit --dataset=TinyImageNet --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
   done
   ```
* Pit on TinyImageNet
   ```shell
     ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
   do
       python multi/ClientTrain.py --client_id=$i --model=pit --dataset=TinyImageNet --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
   done
   ```
**Note:** Please keep the IP addresses of the server and the client consistent. If there are multiple devices running, run the corresponding code directly on the corresponding edge device and replace it with the IP address of the server. The operating instructions of other baselines are in `scripts/difwork`.

### 5.4 Running on ASC
We selected 10 Jetson and rasberry devices with different memory and different computing speeds to test on cifar100, including 2 Jetson-nano devices with 4GB memory, 2 Jetson-Xavier-NX with 16GB memory, 2 Jetson-AgX with 32GB memory and rasberry pi with 4GB memory.
- **Launch the server:**
```shell
## Run on 10 Jetson devices
python multi/server.py --epochs=150 --num_users=10 --frac=0.4 --ip=127.0.0.1:8000
```
**Note：--ip=127.0.0.1:8000 here means that the local machine is used as the center server. If there is an existing server, it can be replaced with the IP address of the server.**

- **Launch the clients:**
* RNN on ASC
    ```shell
   ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
   do
       python multi/ClientTrainNLP.py --client_id=$i --model=rnn --dataset=ASC --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
   done
   ```
* LSTM on ASC
   ```shell
     ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
       python multi/ClientTrainNLP.py --client_id=$i --model=lstm --dataset=ASC --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.0008 --optim=SGD --lr_decay=1e-5 --ip=127.0.0.1:8000
   done
   ```
**Note:** Please keep the IP addresses of the server and the client consistent. If there are multiple devices running, run the corresponding code directly on the corresponding edge device and replace it with the IP address of the server. The operating instructions of other baselines are in `scripts/difwork`.

### 5.5 Running on DSC
We selected 10 Jetson and rasberry devices with different memory and different computing speeds to test on cifar100, including 2 Jetson-nano devices with 4GB memory, 2 Jetson-Xavier-NX with 16GB memory, 2 Jetson-AgX with 32GB memory and rasberry pi with 4GB memory.
- **Launch the server:**
```shell
## Run on 10 Jetson devices
python multi/server.py --epochs=150 --num_users=10 --frac=0.4 --ip=127.0.0.1:8000
```
**Note：--ip=127.0.0.1:8000 here means that the local machine is used as the center server. If there is an existing server, it can be replaced with the IP address of the server.**

- **Launch the clients:**
* bert on DSC
    ```shell
   ## Run on 10 Jetson devices
   for ((i=0;i<10;i++));
   do
       python multi/ClientTrainNLP.py --client_id=$i --model=bert --dataset=DSC --num_classes=100 --task=10 --alg=PuzzleFL --lr=0.001 --optim=Adam --lr_decay=1e-4 --ip=127.0.0.1:8000
   done
   ```
### 5.6 Result
- **The accuracy trend overtime time under different workloads**(X-axis represents the time and Y-axis represents the inference accuracy)
    ![](https://github.com/LINC-BIT/Loci/blob/main/Result.png))


### 6 Citation
The citations of the baseline methods in `baselines/` are listed as follows: 

#### DFL methods:
- GOSSIP
    ```bibtex
    @inproceedings{mcmahan2017communication,
    title={Communication-efficient learning of deep networks from decentralized data},
    author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and y Arcas, Blaise Aguera},
    booktitle={Artificial intelligence and statistics},
    pages={1273--1282},
    year={2017},
    organization={PMLR}
    }

- PENS
    ```bibtex
    @article{onoszko2021decentralized,
    title={Decentralized federated learning of deep neural networks on non-iid data},
    author={Onoszko, Noa and Karlsson, Gustav and Mogren, Olof and Zec, Edvin Listo},
    journal={arXiv preprint arXiv:2107.08517},
    year={2021}
    }

- FedHP 
    ```bibtex
    @inproceedings{liao2023adaptive,
    title={Adaptive configuration for heterogeneous participants in decentralized federated learning},
    author={Liao, Yunming and Xu, Yang and Xu, Hongli and Wang, Lun and Qian, Chen},
    booktitle={IEEE INFOCOM 2023-IEEE Conference on Computer Communications},
    pages={1--10},
    year={2023},
    organization={IEEE}
    }


- HDFL 
    ```bibtex
    @inproceedings{zhang2023novel,
    title={A Novel Hierarchically Decentralized Federated Learning Framework in 6G Wireless Networks},
    author={Zhang, Jie and Chen, Li and Chen, Xiaohui and Wei, Guo},
    booktitle={IEEE INFOCOM 2023-IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS)},
    pages={1--6},
    year={2023},
    organization={IEEE}
    }

#### FCL methods:

- FedWEIT 
    ```bibtex
    @inproceedings{yoon2021federated,
    title={Federated continual learning with weighted inter-client transfer},
    author={Yoon, Jaehong and Jeong, Wonyong and Lee, Giwoong and Yang, Eunho and Hwang, Sung Ju},
    booktitle={International Conference on Machine Learning},
    pages={12073--12086},
    year={2021},
    organization={PMLR}
    }


- FedKNOW
    ```bibtex
    @inproceedings{luopan2023fedknow,
    title={Fedknow: Federated continual learning with signature task knowledge integration at edge},
    author={Luopan, Yaxin and Han, Rui and Zhang, Qinglong and Liu, Chi Harold and Wang, Guoren and Chen, Lydia Y},
    booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
    pages={341--354},
    year={2023},
    organization={IEEE}
    }


- FedViT
    ```bibtex
    @article{zuo2024fedvit,
    title={FedViT: Federated continual learning of vision transformer at edge},
    author={Zuo, Xiaojiang and Luopan, Yaxin and Han, Rui and Zhang, Qinglong and Liu, Chi Harold and Wang, Guoren and Chen, Lydia Y},
    journal={Future Generation Computer Systems},
    volume={154},
    pages={1--15},
    year={2024},
    publisher={Elsevier}
    }
