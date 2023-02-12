# RoboMaster YOLOv5-6.0-keypoints

## 2022/8/3 补丁：

* 优化了背景剔除功能，原以正矩形为边界，现为任意四边形。

## 2022/7/29 补丁：

* 背景剔除时如果仅保留标点四点凸包区域效果不佳，建议尽量保留完整数字或图案。

## 2022/7/18 补丁：

* 添加了红点和大弹丸遮挡的数据增强

## 2022/7/17 补丁：
* 修复了使用mean均值计算最小外接正框中心点的错误

* 四点坐标八个位置同时参与最小外接正框iou损失和关键点回归损失，或会导致一些情况下很难收敛，例如矿石识别和矿石兑换框识别180°旋转增强下的训练（识别整个面，可能问题原因是目标大且少）。而收敛后或能提高回归精度，具体体现为，yolo-face格式下（四个位置负责最小外接正框iou损失，另外八个位置负责关键点回归损失），容易收敛（能做出兑换框（角标）），但点回归并没有本代码准。

## （2022国赛）
主要修改了以下几个部分：

* 修复了不能进行 不选mosaic数据增强 的训练，目的：数据增强能出现更多大目标，结果：mosaic概率参数可调范围[0.0，1.0]
* 所有灰色像素[114，114，114] (出现于数据增强时填充空余区域) 均修改为coco负样本，目的：增加背景环境鲁棒性
* 添加edge_clip操作，将数据集中除装甲板外的背景全部剔除，目的：去除装甲板和车或前哨站本身之类的关联，结果：edge_clip概率参数可调范围[0.0，1.0]
* 删除augmentations.py->random_perspective函数中.clip(0, width)操作，使关键点可超出图像， 目的：适应未来调小roi的操作
* 重新使用了yolov5s尺寸的模型，模型训练收敛更快，连续帧关键点识别更稳定；在AGX及NX上均能稳100帧，NX需有功率模式8(刷机版本4.6.2+)
* 不使用int8部署(精度不正常)

## （2022地区赛）
针对robomaster装甲板自瞄修改后的YOLOv5-6.0目标检测网络，主要修改了以下几个部分：

* 将外接矩形表示的目标修改为用装甲板4个顶点表示目标。修改了loss函数，改成关键点检测。同时修改了数据集的加载方式。
* 将目标分类从多个类别分为一类改成从多个类别分为两类，即颜色分一次类，图案标签分一次类。
* 将COCO数据集作为负样本，加入训练数据中，可以起到降低误识别的作用。
* 使用了yolov5n尺寸的模型
* 不使用int8部署(精度不正常)

* model/yolo.py Detect类中增加设置网络输出尺寸参数(在数据集yaml文件中设置)，结果：废弃cut.py
* export.py 输出两个onnx分别用于Tensorrt部署和openvino部署

**数据集yaml文件**
```shell
    train: /cluster/home/it_stu3/Data/RM-keypoint-4-9-Oplus/images    #图片路径
    val: /cluster/home/it_stu3/Data/RM-keypoint-4-9-Oplus/images      #图片路径

    nc: 36                                                            #class种类数：4*9
    np: 4                                                             #关键点个数
    colors: 4                                                         #颜色种类数
    tags: 9                                                           #标签种类数
    names: ['BG', 'B1', 'B2', 'B3', 'B4', 'B5', 'BO', 'BBs', 'BBb',
            'RG', 'R1', 'R2', 'R3', 'R4', 'R5', 'RO', 'RBs', 'RBb',
            'NG', 'N1', 'N2', 'N3', 'N4', 'N5', 'NO', 'NBs', 'NBb',
            'PG', 'P1', 'P2', 'P3', 'P4', 'P5', 'PO', 'PBs', 'PBb',]  #种类命名

    date: '2022/6/18'                                                 #最后修改日期
```

**训练**
```shell
    python3 train.py 
        --weights ''                                      # 加载预训练模型(不建议使用官方预训练，有cfg就可)
        --cfg 'models/yolov5s.yaml'                       # 网络结构定义
        --data '~/Data/RM-keypoint-4-9-Oplus/armor.yaml'  # 训练数据文件
        --hyp 'data/RM.armor.yaml'                        # 训练参数文件
        --epochs 500                                      # 训练500代(根据实际情况修改)
        --batch-size 16                                   # 单次数据量(报错显存不足则需要降低)
        --img-size 640                                    # 训练图片大小
        --noval                                           # 关闭模型评估(4点模型val的代码没有做修改，用不了)
        --adam                                            # 使用Adam优化器
        --workers 16                                      # 16进程并行加载数据集(根据电脑的CPU量进行修改)
        --negative-path ~/Data/COCO/unlabeled2017/        # 负样本文件夹(支持指定多个负样本文件夹，空格隔开即可)
```
**种类数裁剪(已废弃)**
```shell
    python3 cut.py                                    # 对训练后的模型进行裁剪(修改cut.py中的.pt路径)
```
**测试/识别**
```shell
    python3 detect.py 
        --weights runs/train/exp/weights/best.pt  # 训练后得到的权重文件
        --source ~/data/images                 # 需要识别的视频文件
        --img-size 640                                # 输入网络的图片大小
        --view-img                                    # 实时显示识别结果
```

**模型转化/部署**
```shell
    python3 cut.py                                    #(已废弃) 对训练后的模型进行裁剪(修改cut.py中的.pt路径) 
    python3 models/export.py 
        --weights runs/train/exp/weights/best.pt  # 裁剪后得到的文件
        --img-size 384 640                            # 输入分辨率(a,b)，(a,b)为_trt分辨率；(b,b)为_cv分辨率

    # 到此得到两份onnx文件,最好在部署前能进行onnx-simplify进行简化
    # 其中*_trt.onnx为用于Tensorrt部署的文件，*_cv.onnx为用于openvino部署的文件
```
与YOLOv5-5.0的对比：

* 关键点回归精度更高
* 同尺寸模型，在AGX上进行trt部署后耗时缩减15%
---
---
---
---
---
<div align="center">
<p>
   <a align="left" href="https://ultralytics.com/yolov5" target="_blank">
   <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/splash.jpg"></a>
</p>
<br>
<div>
   <a href="https://github.com/ultralytics/yolov5/actions"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
   <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
   <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
   <br>
   <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
   <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
   <a href="https://join.slack.com/t/ultralytics/shared_invite/zt-w29ei8bp-jczz7QYUmDtgo6r6KcMIAg"><img src="https://img.shields.io/badge/Slack-Join_Forum-blue.svg?logo=slack" alt="Join Forum"></a>
</div>
<br>
<div align="center">
   <a href="https://github.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.linkedin.com/company/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://twitter.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://youtube.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.facebook.com/ultralytics">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="2%"/>
   </a>
   <img width="2%" />
   <a href="https://www.instagram.com/ultralytics/">
   <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="2%"/>
   </a>
</div>

<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<!-- 
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Inference</summary>

Inference with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36). Models automatically download
from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5m, yolov5l, yolov5x, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>



<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading models automatically from
the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

Run commands below to reproduce results
on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on
first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the
largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

<img width="800" src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png">

</details>  

<details open>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️
  RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Roboflow for Datasets, Labeling, and Active Learning](https://github.com/ultralytics/yolov5/issues/4975)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TorchScript, ONNX, CoreML Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

</details>

## <div align="center">Environments</div>

Get started in seconds with our verified environments. Click each icon below for details.

<div align="center">
    <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-colab-small.png" width="15%"/>
    </a>
    <a href="https://www.kaggle.com/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-kaggle-small.png" width="15%"/>
    </a>
    <a href="https://hub.docker.com/r/ultralytics/yolov5">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-docker-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-aws-small.png" width="15%"/>
    </a>
    <a href="https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-gcp-small.png" width="15%"/>
    </a>
</div>  

## <div align="center">Integrations</div>

<div align="center">
    <a href="https://wandb.ai/site?utm_campaign=repo_yolo_readme">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-wb-long.png" width="49%"/>
    </a>
    <a href="https://roboflow.com/?ref=ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-roboflow-long.png" width="49%"/>
    </a>
</div>

|Weights and Biases|Roboflow ⭐ NEW|
|:-:|:-:|
|Automatically track and visualize all your YOLOv5 training runs in the cloud with [Weights & Biases](https://wandb.ai/site?utm_campaign=repo_yolo_readme)|Label and automatically export your custom datasets directly to YOLOv5 for training with [Roboflow](https://roboflow.com/?ref=ultralytics) |


<!-- ## <div align="center">Compete and Win</div>

We are super excited about our first-ever Ultralytics YOLOv5 🚀 EXPORT Competition with **$10,000** in cash prizes!

<p align="center">
  <a href="https://github.com/ultralytics/yolov5/discussions/3213">
  <img width="850" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-export-competition.png"></a>
</p> -->

## <div align="center">Why YOLOv5</div>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/136901921-abcfcd9d-f978-4942-9b97-0e3f202907df.png"></p>
<details>
  <summary>YOLOv5-P5 640 Figure (click to expand)</summary>

<p align="left"><img width="800" src="https://user-images.githubusercontent.com/26833433/136763877-b174052b-c12f-48d2-8bc4-545e3853398e.png"></p>
</details>
<details>
  <summary>Figure Notes (click to expand)</summary>

* **COCO AP val** denotes mAP@0.5:0.95 metric measured on the 5000-image [COCO val2017](http://cocodataset.org) dataset over various inference sizes from 256 to 1536.
* **GPU Speed** measures average inference time per image on [COCO val2017](http://cocodataset.org) dataset using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) V100 instance at batch-size 32.
* **EfficientDet** data from [google/automl](https://github.com/google/automl) at batch size 8.
* **Reproduce** by `python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5n6.pt yolov5s6.pt yolov5m6.pt yolov5l6.pt yolov5x6.pt`
</details>

### Pretrained Checkpoints

[assets]: https://github.com/ultralytics/yolov5/releases
[TTA]: https://github.com/ultralytics/yolov5/issues/303

|Model |size<br><sup>(pixels) |mAP<sup>val<br>0.5:0.95 |mAP<sup>val<br>0.5 |Speed<br><sup>CPU b1<br>(ms) |Speed<br><sup>V100 b1<br>(ms) |Speed<br><sup>V100 b32<br>(ms) |params<br><sup>(M) |FLOPs<br><sup>@640 (B)
|---                    |---  |---    |---    |---    |---    |---    |---    |---
|[YOLOv5n][assets]      |640  |28.4   |46.0   |**45** |**6.3**|**0.6**|**1.9**|**4.5**
|[YOLOv5s][assets]      |640  |37.2   |56.0   |98     |6.4    |0.9    |7.2    |16.5
|[YOLOv5m][assets]      |640  |45.2   |63.9   |224    |8.2    |1.7    |21.2   |49.0
|[YOLOv5l][assets]      |640  |48.8   |67.2   |430    |10.1   |2.7    |46.5   |109.1
|[YOLOv5x][assets]      |640  |50.7   |68.9   |766    |12.1   |4.8    |86.7   |205.7
|                       |     |       |       |       |       |       |       |
|[YOLOv5n6][assets]     |1280 |34.0   |50.7   |153    |8.1    |2.1    |3.2    |4.6
|[YOLOv5s6][assets]     |1280 |44.5   |63.0   |385    |8.2    |3.6    |16.8   |12.6
|[YOLOv5m6][assets]     |1280 |51.0   |69.0   |887    |11.1   |6.8    |35.7   |50.0
|[YOLOv5l6][assets]     |1280 |53.6   |71.6   |1784   |15.8   |10.5   |76.8   |111.4
|[YOLOv5x6][assets]<br>+ [TTA][TTA]|1280<br>1536 |54.7<br>**55.4** |**72.4**<br>72.3 |3136<br>- |26.2<br>- |19.4<br>- |140.7<br>- |209.8<br>- 

<details>
  <summary>Table Notes (click to expand)</summary>

* All checkpoints are trained to 300 epochs with default settings and hyperparameters.
* **mAP<sup>val</sup>** values are for single-model single-scale on [COCO val2017](http://cocodataset.org) dataset.<br>Reproduce by `python val.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`
* **Speed** averaged over COCO val images using a [AWS p3.2xlarge](https://aws.amazon.com/ec2/instance-types/p3/) instance. NMS times (~1 ms/img) not included.<br>Reproduce by `python val.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`
* **TTA** [Test Time Augmentation](https://github.com/ultralytics/yolov5/issues/303) includes reflection and scale augmentations.<br>Reproduce by `python val.py --data coco.yaml --img 1536 --iou 0.7 --augment`

</details>

## <div align="center">Contribute</div>

We love your input! We want to make contributing to YOLOv5 as easy and transparent as possible. Please see
our [Contributing Guide](CONTRIBUTING.md) to get started, and fill out
the [YOLOv5 Survey](https://ultralytics.com/survey?utm_source=github&utm_medium=social&utm_campaign=Survey) to provide 
thoughts and feedback on your experience with YOLOv5. Thank you!

## <div align="center">Contact</div>

For issues running YOLOv5 please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues). For business or
professional support requests please visit [https://ultralytics.com/contact](https://ultralytics.com/contact).

<br>

<div align="center">
    <a href="https://github.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.linkedin.com/company/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://twitter.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-twitter.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://youtube.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-youtube.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.facebook.com/ultralytics">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-facebook.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.instagram.com/ultralytics/">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-instagram.png" width="3%"/>
    </a>
</div>
