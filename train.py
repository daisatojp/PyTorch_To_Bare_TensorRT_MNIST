import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


class MyNet(torch.nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(800, 500)
        self.fc2 = torch.nn.Linear(500, 10)

    def forward(self, x):
        """
        :param x: (B, 1, 28, 28)
        """
        h = self.conv1(x)  # (B, 20, 24, 24)
        h = self.pool1(h)  # (B, 20, 12, 12)
        h = self.conv2(h)  # (B, 50, 8, 8)
        h = self.pool2(h)  # (B, 50, 4, 4)
        h = h.reshape(-1, 800)  # (B, 800)
        h = self.fc1(h)  # (B, 500)
        h = torch.relu(h)  # (B, 500)
        h = self.fc2(h)  # (B, 10)
        return f.log_softmax(h, dim=1)


def main():
    device = 'cuda:0'

    epoch = 20
    batch = 128
    intensity = 1.0

    net = MyNet()
    net = net.to(device)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * intensity)
            ])),
        batch_size=batch,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            './data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * intensity)
            ])),
        batch_size=batch,
        shuffle=True)
    optimizer = torch.optim.Adam(
        params=net.parameters(), lr=0.001)

    for e in range(epoch):
        loss = None
        net.train(True)
        for i, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = net(data)
            loss = f.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f'Training log: {e+1} epoch ({(i+1)*128} / 60000 train. data). Loss: {loss.item()}')
        net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(device)
                target = target.to(device)
                output = net(data)
                test_loss += f.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= 10000
        print(f'Test loss (avg): {test_loss}, Accuracy: {correct / 10000}')

    torch.save(net.state_dict(), 'checkpoint.pt')
    script = torch.jit.trace(net, torch.ones(size=(1, 1, 28, 28), dtype=torch.float32, device=device))
    torch.jit.save(script, 'script.pt')


if __name__ == '__main__':
    main()
