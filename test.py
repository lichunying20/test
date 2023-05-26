import argparse
import time
import json
import os

from tqdm import tqdm
from models.resnet_main import *
# from efficientnet_pytorch import EfficientNet
from torch import nn
from torch import optim
# from torch.optim.lr_scheduler import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tools import warmup_lr


# 初始化参数
def get_args():
    """在下面初始化你的参数.
    """
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
    parser.add_argument('--model', type=str, default='ResNet34')

    # scheduler
    parser.add_argument('--warmup_epoch', type=int, default=1)

    # 返回参数集
    return parser.parse_args()


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
        test_dataset = datasets.ImageFolder(
            self.opt.test_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.opt.data_mean, self.opt.data_std)  # 添加归一化处理，与train.py保持一致
            ])
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.opt.test_batch_size,
            shuffle=False,
            **kwargs
        )
        # 挑选神经网络、参数初始化
        net = ResNet34()
        assert net is not None

        self.model = net.to(self.device)

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()

    def test(self):
        self.model.eval()
        testing_loss = 0
        num_correct = 0
        with torch.no_grad():
            bar = tqdm(self.test_loader)
            for data, target in bar:
                # 测试中...
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                testing_loss += self.loss_function(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                num_correct += pred.eq(target.view_as(pred)).sum().item()
            bar.close()

        # 打印测试结果
        testing_loss /= len(self.test_loader)
        print('test >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
            testing_loss,
            num_correct,
            len(self.test_loader.dataset),
            100. * num_correct / len(self.test_loader.dataset))
        )


if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)
    args = get_args()
    tester = Worker(args=args)

    # 测试
    tester.test()
