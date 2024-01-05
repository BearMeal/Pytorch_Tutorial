import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
print('hi')

# 파라미터 설정하기
lr = 1e-3
batch_size = 64
num_epoch = 10

# 학습가중치와 텐서보드 로그저장
ckpt_dir = './checkpoint'
log_dir = './log'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 네트워크 구축
class Net(nn.Module):
    def __init__(self):
        # 넷의 부모클래스에서 속성 상속받아옴
        super(Net, self).__init__()
        
        # 네트워크 구축에 필요한 레이어 초기화 
        self.conv1 = nn.Conv2d(1, 10, 5, 1, 0, 0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(10, 20, 5, 1, 0, 0)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        
        self.fc1 = nn.Linear(320, 50)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout2d(0.5)
        
        self.fc1 = nn.Linear(50, 10)
        
    # 위의 초기화된 레이어를 연결하여 네트워크를 구축하였다
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        
        # 텐서의 1차원은 자동, 2차원은 320 크기로
        # 개수가 6400개이면 앞은 20이 되어야함 그런데 -1하면 알아서 맞춰줌
        x = x.view(-1, 320)
        
        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)
        
        x = self.fc2(x)
        
        return x
        
        
# 네트워크를 저장 또는 불러오는 함수
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
                   './%s/model_epoch%d.pth' % (ckpt_dir, epoch))
        
def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort()
    
    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))
    
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    return net, optim

## mnist데이터 불러오기
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
dataset = datasets.MNIST(download=True, root='./', train=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

num_data = len(loader.dataset)
num_batch = np.ceil(num_data / batch_size)

## 네트워크 설정 및 필요한 손실함수 구현
net = Net().to(device)
params = net.parameters()

fn_loss = nn.CrossEntropyLoss().to(device)
fn_pred = lambda output: torch.softmax(output, dim=1)
fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()
optim = torch.optim.Adam(params, lr=lr)
writer = SummaryWriter(log_dir=log_dir)

## 네트워크 학습 시작
for epoch in range(1, num_epoch+1):
    net.train()
    loss_arr = []
    acc_arr = []
    
    for batch, (input, label) in enumerate(loader, 1):
        input = input.to(device)
        label = label.to(device)
        output = net(input)
        pred = fn_pred(output)
        optim.zero_grad()
        loss = fn_loss(output, label)
        acc = fn_acc(pred, label)
        loss.backward()
        optim.step()
        loss_arr += [loss.item()]
        acc_arr += [acc.item()]
        print('TRAIN: EPOCH %04d/%04d | BATCH %04d/$04d | LOSS: %.4f | ACC: %.4f' %
              (epoch, num_epoch, num_batch, np.mean(loss_arr), np.mean(acc_arr)))
        
    writer.add_scalar('loss', np.mean(loss_arr), epoch)
    writer.add_scalar('acc', np.mean(acc_arr), epoch)
    
    save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)
writer.close()
        
        


