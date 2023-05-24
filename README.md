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

## test代码(模仿train.py)
```python

```

## test所用的模型代码（ResNet34模型）
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 把残差连接补充到 Block 的 forward 函数中
class Block(nn.Module):
    def __init__(self, dim, out_dim, stride) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(dim, out_dim, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # 使用self.conv1对x进行卷积操作
        x = self.bn1(x)    # 使用self.bn1对卷积结果进行批量归一化操作
        x = self.relu1(x)  # 对批量归一化结果进行ReLU激活函数操作
        x = self.conv2(x)  # 使用self.conv2对上一层的结果进行卷积操作
        x = self.bn2(x)    # 使用self.bn2对卷积结果进行批量归一化操作
        x = self.relu2(x)  # 对批量归一化结果进行ReLU激活函数操作
        return x

class ResNet32(nn.Module):
    def __init__(self, in_channel=64, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = in_channel

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3)
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        self.last_channel = in_channel

        self.layer1 = self._make_layer(in_channel=64, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(in_channel=128, num_blocks=4, stride=2)
        self.layer3 = self._make_layer(in_channel=256, num_blocks=6, stride=2)
        self.layer4 = self._make_layer(in_channel=512, num_blocks=3, stride=2)

        self.avgpooling = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Linear(4608, self.num_classes)

    def _make_layer(self, in_channel, num_blocks, stride):
        layer_list = [Block(self.last_channel, in_channel, stride)]
        self.last_channel = in_channel
        for i in range(1, num_blocks):
            b = Block(in_channel, in_channel, stride=1)
            layer_list.append(b)
        return nn.Sequential(*layer_list)

    def forward(self, x):   # bs 表示批次大小，64 表示通道数，56 表示高度，56 表示宽度。
        x = self.conv1(x)   # [bs, 64, 56, 56] 特征提取过程 这是我调试出来的结果：[bs, 64, 112, 112]
        x = self.maxpooling(x)  # [bs, 64, 28, 28]池化，降低分辨率和计算量 这是我调试出来的结果：[bs,64,56,56]
        x = self.layer1(x)  # [bs,64,56,56]，对输入的特征图 x 进行卷积和池化操作，并将其传递给模型的第一层卷积块 layer1 进行处理。
        x = self.layer2(x)  # [bs,128,28,28]，将 layer1 的输出作为输入，传递给模型的第二层卷积块 layer2 进行处理。进行卷积和池化。
        x = self.layer3(x)  # [bs,256,14,14]，将 layer2 的输出作为输入，传递给模型的第三层卷积块 layer3 进行处理。进行卷积和池化。
        x = self.layer4(x)  # [bs,512,7,7]，将 layer3 的输出作为输入，传递给模型的第四层卷积块 layer4 进行处理。进行卷积和池化。
        x = self.avgpooling(x)  # [bs,512,3,3]，对输入的特征图 x 进行平均池化操作，将每个通道的特征图缩小为一个标量值。
        x = x.view(x.shape[0], -1)  # [bs,4608]，将 [bs, 512, 7, 7] 的特征图展平为一个形状为 [bs, 4608] 的向量。bs为8。
        x = self.classifier(x)  # [bs,2]，该层的输出形状为 [bs, 2]，其中 bs 表示批次大小，2 表示输出的类别数目。bs为8。
        output = F.softmax(x)

        return output

if __name__=='__main__':
    t = torch.randn([8, 3, 224, 224])
    model = ResNet32()
    out = model(t)
    print(out.shape)
```
## 最后得到的结果
![image](https://github.com/lichunying20/test/assets/128216499/a4422295-6ef4-4520-b3b2-5dce7cd5477c)

