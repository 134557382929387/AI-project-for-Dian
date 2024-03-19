import torch
import  torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#导入数据集、加载数据
train_set = torchvision.datasets.MNIST(root="MNIST",train=True,transform= torchvision.transforms.ToTensor(),download=True)
test_set = torchvision.datasets.MNIST(root="MNIST",train=False,transform= torchvision.transforms.ToTensor(),download=True)
train_dataloader = DataLoader(train_set,32,shuffle=False,num_workers=0)
test_dataloader = DataLoader(test_set,32,shuffle=False,num_workers=0)
#神经网络结构定义
class HHH(torch.nn.Module):
    def __init__(self):
        super(HHH,self).__init__()
        # self.model =nn.Sequential(
        #     nn.Conv2d(1,4,5),#4*24*24
        #     nn.MaxPool2d(2),#4*12*12
        #     nn.Conv2d(4,8,5),#8*8*8
        #     nn.MaxPool2d(2),#8*4*4
        #     nn.Flatten(),
        #     nn.Linear(8*4*4,256),#?
        #     nn.ReLU(),
        #     nn.Linear(256,10)
        # )
        self.conv2d1 = nn.Conv2d(1,4,5)#4*24*24
        self.conv2d2 = nn.Conv2d(4,8,5)#8*8*8
        self.maxpool = nn.MaxPool2d(2)#4*12*12
        self.linear1 = nn.Linear(8*4*4,256)#?
        self.linear2 =  nn.Linear(256,10)

    def forward(self,x):
        x=self.conv2d1(x)
        x=self.maxpool(x)
        x = self.conv2d2(x)
        x = self.maxpool(x)
        x = nn.Flatten()(x)#要以函数形式调用Relu
        x = self.linear1(x)
        x = nn.ReLU()(x)#要以函数形式调用Relu
        x = self.linear2(x)
        # x=self.model(x)
        return x

class Statis:#指标统计函数
    def Accuracy(self, outputs, targets):
        correct = (outputs.argmax(1) == targets).sum().item()
        accuracy = correct / len(targets)
        return accuracy

    def Recall(self, outputs, targets):
        recall = 0
        for i in range(10):
            positive = (targets == i).sum().item()
            correct_prediction = ((outputs == i) & (targets == i)).sum().item()
            if positive != 0:
                rate = correct_prediction / positive
                recall += rate
        return recall / 10  # 取平均值作为最终结果

    def Precision(self, outputs, targets):
        precision = 0
        for i in range(10):
            positive = (outputs == i).sum().item()
            correct_prediction = ((outputs == i) & (targets == i)).sum().item()
            if positive != 0:
                rate = correct_prediction / positive
                precision += rate
        return precision / 10  # 取平均值作为最终结果

    def F1(self, outputs, targets):
        p = self.Precision(outputs, targets)
        r = self.Recall(outputs, targets)
        if p + r == 0:
            return 0
        else:
            return 2 * p * r / (p + r)


writer = SummaryWriter("linear_train")
hhh = HHH()
statis = Statis()
#损失函数
loss_fuc=nn.CrossEntropyLoss()
#优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(hhh.parameters(),lr = learning_rate)#parameters为网络的可学习参数

#初始状态
total_train_step = 0
total_correct = 0
for data in test_dataloader:
    imgs, targets = data
    outputs = hhh(imgs)
    loss = loss_fuc(outputs, targets)
    # total_test_loss = total_test_loss + loss.item()
print(f"初始测试正确率：{statis.Accuracy(outputs,targets)}")
writer.add_scalar("accuracy",total_correct / len(test_set),global_step=1)
step=1

for echo in range(20):
    cost = 0#成本函数    # 开始训练

    hhh.train()
    for data in train_dataloader:
        imgs, targets = data
        output = hhh(imgs)
        loss=loss_fuc(output, targets)
        cost = cost +loss

        #
        total_train_step = total_train_step + 1
        # if total_train_step % 1000 == 0:
        #     print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #开始测试
    hhh.eval()
    total_test_loss = 0
    total_correct = 0
    with torch.no_grad():#
        for data in test_dataloader:
            imgs, targets = data
            outputs = hhh(imgs)
            loss = loss_fuc(outputs, targets)
            total_test_loss = total_test_loss + loss.item()

    print(f"第{echo + 1}轮测试正确率：{statis.Accuracy(outputs,targets)}")
    step = step + 1
    writer.add_scalar("accuracy", statis.Accuracy(outputs,targets), global_step=step)

writer.close()
    # print(f"第{echo+1}轮训练，loss：{cost}")




