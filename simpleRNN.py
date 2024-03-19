import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载数据集
transforms = torchvision.transforms.ToTensor()
train_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=True, transform=transforms, download=True)
test_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=False, transform=transforms, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.tanh(self.i2h(combined))
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

loss_func = torch.nn.CrossEntropyLoss()
rnn = SimpleRNN(28, 128, 10)  # 修改隐藏层大小为128，使得模型能够更好地捕获信息
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)

writer = SummaryWriter("SimpleRNN_test")
step = 0
for epoch in range(50):
    rnn.train()  # 进入训练模式
    total_correct = 0
    total_loss = 0
    for i, (imgs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        hidden = rnn.initHidden(imgs.size(0))  # 每个batch开始时重新初始化隐藏状态
        imgs = imgs.view(-1, 28, 28)  # 调整输入图片的形状
        #RNN循环过程
        for j in range(imgs.size(1)):
            output, hidden = rnn(imgs[:, j, :], hidden)

        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += (output.argmax(1) == labels).sum().item()

    average_loss = total_loss / len(train_dataloader)
    accuracy = total_correct / len(train_dataset)
    print(f"第{epoch + 1}轮：训练损失：{average_loss}, 训练正确率：{accuracy}")

    writer.add_scalar("train_loss", average_loss, global_step=epoch)
    writer.add_scalar("train_accuracy", accuracy, global_step=epoch)

    # 在测试之前要将模型设为评估模式
    rnn.eval()
    total_correct = 0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(test_dataloader):
            hidden = rnn.initHidden(imgs.size(0))  # 每个batch开始时重新初始化隐藏状态
            imgs = imgs.view(-1, 28, 28)
            for j in range(imgs.size(1)):
                output, hidden = rnn(imgs[:, j, :], hidden)
            total_correct += (output.argmax(1) == labels).sum().item()

    accuracy = total_correct / len(test_dataset)
    print(f"第{epoch + 1}轮：测试正确率：{accuracy}")

    writer.add_scalar("test_accuracy", accuracy, global_step=epoch)

writer.close()
