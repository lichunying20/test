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

        test_dataset = datasets.ImageFolder(
            args.test_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        self.test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )

        # 挑选神经网络、参数初始化
        net = ResNet34()
        assert net is not None

        self.model = net.to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.lr
        )

        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()

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
        test_acc, test_loss = worker.test()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, test_acc, test_loss)
            torch.save(worker.model, save_dir)