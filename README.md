# test
## 验证集和测试集的相同点和目的(The similarities and purposes of validation set and test set.)

相同点：验证集和测试集都是从原始数据集中分离出来的，用于评估模型的泛化能力和性能。

Commonalities: The validation set and the test set are both separated from the original dataset and used to evaluate the model's generalization ability and performance.

目的：验证集和测试集的目的都是评估模型的性能和泛化能力，以便在实际应用中选择最佳模型。

Purpose: The purpose of both the validation set and the test set is to evaluate the performance and generalization ability of the model, in order to select the best model for practical applications.

## train代码
```python
import ...

# 初始化参数
def get_args():...
    # model
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=[
                            'ResNet18',
                            'ResNet34',
                            'ResNet50',
                            'ResNet18RandomEncoder',
                        ])

    # scheduler
    parser.add_argument('--warmup_epoch', type=int, default=1)

    # 通过json记录参数配置
    args = parser.parse_args()
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    log_file = os.path.join(args.directory, 'log.json')
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    with open(log_file, 'w') as log:
        json.dump(vars(args), log, indent=4)

    # 返回参数集
    return args

class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')
        kwargs = {
            'num_workers': args.num_workers,
            'pin_memory': True,
        } if args.is_cuda else {}

        # 载入数据
        train_dataset = datasets.ImageFolder(
            ...
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        self.train_loader = DataLoader(
            ...
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        # 挑选神经网络、参数初始化
       ...

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()

        # warm up 学习率调整部分
        self.per_epoch_size = len(train_dataset) // args.batch_size
        self.warmup_step = args.warmup_epoch * self.per_epoch_size
        self.max_iter = args.epochs * self.per_epoch_size
        self.global_step = 0

    def train(...):...

    def val(self):
        self.model.eval()
        validating_loss = 0
        num_correct = 0
        with torch.no_grad():
            bar = tqdm(self.val_loader)
            for data, target in bar:
                # 测试中...
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validating_loss += self.loss_function(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                num_correct += pred.eq(target.view_as(pred)).sum().item()
            bar.close()

        # 打印验证结果
        validating_loss /= len(self.val_loader)
        ...

        # 返回重要信息，用于生成模型保存命名
        return 100. * num_correct / len(self.val_loader.dataset), validating_loss

if __name__ == '__main__':
    # 初始化
    ...
    # 训练与验证
    for epoch in range(1, args.epochs + 1):...
```
## 运行train.py后得到的结果
![image](https://github.com/lichunying20/test/assets/128216499/88f2b551-df4c-4050-ad70-54ec89e25420)

## 将train.py(ResNet18模型改为ResNet34模型)
```python
from tqdm import tqdm
from models import *

# model
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=[
                            'ResNet18',
                            'ResNet34',
                            'ResNet50',
                            'ResNet18RandomEncoder',
                        ])
# 载入数据
        train_dataset = datasets.ImageFolder(
            args.train_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
                        
  # 挑选神经网络、参数初始化
        net = None
        if args.model == 'ResNet18':
            net = ResNet18(num_cls=args.num_classes)
        elif args.model == 'ResNet34':
            net = ResNet34(num_cls=args.num_classes)
        elif args.model == 'ResNet50':
            net = ResNet50(num_cls=args.num_classes)
        elif args.model == 'ResNet18RandomEncoder':
            net = ResNet18RandomEncoder(num_cls=args.num_classes)
        assert net is not None
 ```
 改为
 ```python
 
from tqdm import tqdm
from models.resnet_main import *

 # model
    parser.add_argument('--model', type=str, default='ResNet34')
                        
 # 载入数据
        train_dataset = datasets.ImageFolder(
            args.train_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
 # 挑选神经网络、参数初始化
       net = ResNet34()
        assert net is not None
 ``` 
 
## 运行train.py后得到的结果（ResNet34模型）
![image](https://github.com/lichunying20/test/assets/128216499/ca507098-afaa-447c-a902-2eaf879e14ce)


## test代码(模仿train.py)（ResNet34模型）
```python
import argparse
import time
import json
import os

from tqdm import tqdm
from models import *
from torch import nn
from torch import optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tools import warmup_lr

# 初始化参数
def get_args():
    """在下面初始化你的参数."""
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')
    # exp
    parser.add_argument('--time_exp_start', type=str,
                        default=time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--test_dir', type=str, default='data/test')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_station', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_mps', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    # dataset
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_mean', type=tuple, default=[.5, .5, .5])
    parser.add_argument('--data_std', type=tuple, default=[.5, .5, .5])
    # model
    parser.add_argument('--model', type=str, default='ResNet34(1)',
                        choices=[
                            'ResNet18',
                            'ResNet34',
                            'ResNet50',
                            'ResNet18RandomEncoder',
                            'ResNet34(1)',
                        ])
    # scheduler
    parser.add_argument('--warmup_epoch', type=int, default=1)

    # 通过json记录参数配置
    args = parser.parse_args()
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    log_file = os.path.join(args.directory, 'log.json')
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    with open(log_file, 'w') as log:
        json.dump(vars(args), log, indent=4)

    # 返回参数集
    return args

class Worker:
    def __init__(self, args):
        self.opt = args

        # 判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu')
        kwargs = {
            'num_workers': args.num_workers,
            'pin_memory': True,
        } if args.is_cuda else {}

        # 载入数据
        train_dataset = datasets.ImageFolder(
            args.train_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        test_dataset = datasets.ImageFolder(
            args.test_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        # 挑选神经网络、参数初始化
        net = None
        if args.model == 'ResNet18':
            net = ResNet18(num_cls=args.num_classes)
        elif args.model == 'ResNet34':
            net = ResNet34(num_cls=args.num_classes)
        elif args.model == 'ResNet50':
            net = ResNet50(num_cls=args.num_classes)
        elif args.model == 'ResNet18RandomEncoder':
            net = ResNet18RandomEncoder(num_cls=args.num_classes)
        elif args.model == 'ResNet34(1)':
            net = ResNet34(num_cls=args.num_classes)
        assert net is not None

        self.model = net.to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()

        # warm up 学习率调整部分
        self.per_epoch_size = len(train_dataset) // args.batch_size
        self.warmup_step = args.warmup_epoch * self.per_epoch_size
        self.max_iter = args.epochs * self.per_epoch_size
        self.global_step = 0

    def train(self, epoch):
        self.model.train()
        bar = tqdm(enumerate(self.train_loader))
        for batch_idx, (data, target) in bar:
            self.global_step += 1
            data, target = data.to(self.device), target.to(self.device)

            # 训练中...
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            lr = warmup_lr.adjust_learning_rate_cosine(
                self.optimizer, global_step=self.global_step,
                learning_rate_base=self.opt.lr,
                total_steps=self.max_iter,
                warmup_steps=self.warmup_step
            )

            # 更新进度条
            bar.set_description(
                'train epoch {} >> [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f} >> '.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item(),
                    lr
                )
            )
        bar.close()

    def test(self):
        self.model.eval()
        test_loss = 0
        num_correct = 0
        with torch.no_grad():
            bar = tqdm(self.test_loader)
            for data, target in bar:
                # 测试中...
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_function(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                num_correct += pred.eq(target.view_as(pred)).sum().item()
            bar.close()

        # 打印验证结果
        test_loss /= len(self.test_loader)
        print('test >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
            test_loss,
            num_correct,
            len(self.test_loader.dataset),
            100. * num_correct / len(self.test_loader.dataset))
        )

        # 返回重要信息，用于生成模型保存命名
        return 100. * num_correct / len(self.test_loader.dataset), test_loss

if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)
    args = get_args()
    worker = Worker(args=args)

    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        worker.train(epoch)
        test_acc, test_loss = worker.test()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, test_acc, test_loss)
            torch.save(worker.model, save_dir)
```
## 运行test.py后得到的结果（ResNet34模型）
![image](https://github.com/lichunying20/test/assets/128216499/e51e6575-02ae-44e8-9232-b9f68fe3e7ef)

## 总结（summarize）
1、所用的ResNet34（1）模型要与train文件中resnet包括的ResNet34区分开。

The ResNet34 (1) model used should be distinguished from ResNet34 included in the train file.

2、搭建test要参考val，按照val的步骤一步一步的搭建。

Setting up the test should refer to the val file and follow the step-by-step process for validation.

3、了解了验证集和测试集的不同点和相同点以及它们的目的。

Understanding the differences and similarities between the validation set and the test set, as well as their purposes.

不同点：

（1）、验证集用于调整模型的超参数，例如学习率、正则化系数等。通过在验证集上评估不同超参数的性能，可以选择最佳超参数来优化模型。

The validation set is used to adjust the model's hyperparameters, such as learning rate and regularization coefficient. By evaluating the performance of different 
hyperparameter combinations on the validation set, the optimal hyperparameters can be selected to optimize the model.

（2）、测试集用于最终评估模型的性能。在模型开发过程中，应该避免使用测试集来调整模型或选择超参数，否则可能导致过拟合测试集，使得模型在真实场景中的性能表现不佳。

The test set is used for the final evaluation of the model's performance. During model development, it should be avoided to use the test set to adjust the model 
or select hyperparameters, as it may lead to overfitting to the test set and poor performance in real-world scenarios.

（3）、验证集通常是从训练集中划分出来的一部分数据，用于训练集内部的模型选择。而测试集通常是从与训练集不同的数据集中抽样，用于评估模型在真实场景下的性能。

The validation set is typically a subset of the training set, used for internal model selection within the training set. The test set, on the other hand, is 
typically sampled from a different dataset than the training set, used to evaluate the model's performance in real-world scenarios.

（4）、验证集的数据量通常要比测试集小，因为验证集需要多次使用来评估不同的超参数组合。而测试集的数据量应该足够大，以确保模型在真实场景下的性能得到充分评估。

The validation set usually has a smaller sample size compared to the test set because it needs to be used multiple times to evaluate different hyperparameter 
combinations. The test set, however, should have a sufficiently large sample size to ensure that the model's performance in real-world scenarios is adequately 
evaluated.
