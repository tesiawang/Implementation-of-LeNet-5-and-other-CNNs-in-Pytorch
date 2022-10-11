import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import numpy as np
import matplotlib.pyplot as plt
torch.__version__

# hyper-parameters
def add_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--EPOCHS', type = int, default = 20)
    parser.add_argument('--BATCH_SIZE', type = int, default= 512)
    parser.add_argument('--LEARNING_RATE', type = float, default=0.001)
    args = parser.parse_args()
    return args

args = add_args()

BATCH_SIZE = args.BATCH_SIZE
EPOCHS = args.EPOCHS
LEARNING_RATE = args.LEARNING_RATE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, 
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


# show figures
# train_features, train_labels = next(iter(train_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# print(label)
# plt.imshow(img, cmap="gray")
# plt.savefig('new_train_set_fig_0.png')


# img = train_features[1].squeeze()
# label = train_labels[1]
# print(label)
# plt.imshow(img, cmap="gray")
# plt.savefig('new_train_set_fig_1.png')



test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=True)


# show figures
# test_features, test_labels = next(iter(train_loader))
# img = test_features[0].squeeze()
# label = test_labels[0]
# plt.imshow(img, cmap="gray")
# plt.savefig('new_test_set_fig_0.png')


# img = test_features[1].squeeze()
# label = test_labels[1]
# plt.imshow(img, cmap="gray")
# plt.savefig('new_test_set_fig_1.png')


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1,28x28
        self.conv1=nn.Conv2d(1,10,5) # 10, 24x24
        self.conv2=nn.Conv2d(10,20,3) # 128, 10x10
        self.fc1 = nn.Linear(20*10*10,500)
        self.fc2 = nn.Linear(500,10)
    def forward(self,x):
        in_size = x.size(0)
        out = self.conv1(x) #24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  #12
        out = self.conv2(out) #10
        out = F.relu(out)
        out = out.view(in_size,-1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.log_softmax(out,dim=1)
        return out

class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x,dim=1)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        return np.prod(size)


model = LeNet().to(DEVICE)
print(model)
optimizer = optim.Adam(model.parameters(),lr = LEARNING_RATE)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        
        pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
        correct += pred.eq(target.view_as(pred)).sum().item()
    
    print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    
    train_loss = loss.item()
    train_acc = 100. * correct / len(train_loader.dataset)

    return train_loss, train_acc


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_loss, test_acc

rec_train_loss = []
rec_train_acc = []
rec_test_loss = []
rec_test_acc = []

for epoch in range(1, EPOCHS + 1):
    [train_loss, train_acc] = train(model, DEVICE, train_loader, optimizer, epoch)
    [test_loss, test_acc] = test(model, DEVICE, test_loader)
    rec_train_loss.append(train_loss)
    rec_train_acc.append(train_acc)
    rec_test_loss.append(test_loss)
    rec_test_acc.append(test_acc)

np.save('EP{}_LR{}_BS{}_train_loss.npy'.format(EPOCHS,LEARNING_RATE,BATCH_SIZE), rec_train_loss)
np.save('EP{}_LR{}_BS{}_train_acc.npy'.format(EPOCHS,LEARNING_RATE,BATCH_SIZE), rec_train_acc)
np.save('EP{}_LR{}_BS{}_test_loss.npy'.format(EPOCHS,LEARNING_RATE,BATCH_SIZE), rec_test_loss)
np.save('EP{}_LR{}_BS{}_test_acc.npy'.format(EPOCHS,LEARNING_RATE,BATCH_SIZE), rec_test_acc)




