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

## 将train.py

## 运行train.py后得到的结果（ResNet34（1）模型）
![image](https://github.com/lichunying20/test/assets/128216499/7ee8bdb9-0bae-4edb-96eb-fdb3a686111d)


## test代码(模仿train.py)（ResNet34（1）模型）
```python

```


