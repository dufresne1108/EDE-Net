
### 介绍
虽然人类有能力学习新任务的同时不忘记以前学过的知识,但神经网络却很难做到这一点,因为其所学的知识是以参数的形式进行存储,一旦学习新任务将不可避免的把已学习的旧知识参数覆盖掉,造成著名的灾难性遗忘现象.
本文提出一种新颖的高效率动态扩张网络(Efficiently Dynamic Expansion Network,简称EDE-Net),
该网络使用两阶段训练方式高效率地扩张模型:在第一阶段中,针对新来的任务数据集,构建临时模型并利用无标签数据与当前任务数据进行混合训练,然后逐步压缩模型,将其添加到EDE-Net子模块中;
在第二阶段,将当前子模块与已训练的子模型在输出层进行拼接,并利用少量的回溯样本对EDE-Net进行平衡微调训练.
广泛的实验结果表明, EDE-Net仅需要少量的参数和回溯样本就能达到较高的性能.

### 实现细节 
我们的模型模型通过PyTorch代码实现,使用NVIDIA1080Ti作为实验环境.对于每一阶段的训练,我们使用ResNet-18作为基础特征提取器.
我们使用Adam优化器与余弦退火(cosine annealing)学习率衰减策略训练模型,起始学习率为5e-4,权重衰减率为5e-4.数据增强策略则使用常规的随机剪裁,翻转和旋转等操作.
在特征提取器的通道裁剪阶段,为了不破坏残差模块的结构,我们取残差模块所有卷积通道的平均值重构网络,然后使用知识蒸馏训练压缩网络.


#### CIFAR100增量学习实验

1.训练CIFAR100的各阶段特征提取器,可运行train_classifier.py中以下代码：
split_type="s50-t10"表示其实学习50分类，剩余分类10阶段训练.

train_model(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                split_type="s0-t10", prune=True, mixmatch=True)
                
train_model(split_list=[0, 20, 40, 60, 80, 100],
 split_type="s0-t5", mixmatch=True)

train_model(split_list=[0, 50, 60, 70, 80, 90, 100],
 split_type="s50-t5", mixmatch=True)

train_model(split_list=[0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
 split_type="s50-t10", mixmatch=True)

```sh
python train_classifier.py train_model(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                split_type="s0-t10", prune=True, mixmatch=True)
```

2.训练EDE-Net

train_model(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                split_type="s0-t10", prune=True, rehearsal_size=2000)
                
train_model(split_list=[0, 20, 40, 60, 80, 100], 
split_type="s0-t5", rehearsal_size=2000)

train_model(split_list=[0, 50, 60, 70, 80, 90, 100],
 split_type="s50-t5", rehearsal_size=2000)
 
train_model(split_list=[0, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
 split_type="s50-t10", rehearsal_size=2000)


```sh
python train_EDE_Net.py train_model(split_list=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],split_type="s0-t10", prune=True, rehearsal_size=2000)
```
#### 实验(MNIST->SVNH->cifar10)持续学习
1.训练分类器
```sh
python trainMTIL.py train_MTIL_classifier(data_type="MNIST",batch_size=128)

python trainMTIL.py train_MTIL_classifier(data_type="SVNH",batch_size=128)

python trainMTIL.py train_MTIL_classifier(data_type="cifar10",batch_size=128)
```

2.训练 EDE-Net (MNIST->SVNH->cifar10)

```sh
python trainMTIL.py train_EDE_Net(data_list=["MNIST", "SVNH", 'cifar10'], num_per_class=5)
```

