import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, test_loader
from cnn import CNN


if __name__ == '__main__':
    net = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    total_epochs = 2
    total_iters = len(train_loader)
    print_interval = 1000
    
    print('Starting training...')
    for epoch in range(total_epochs):
        avg_loss = 0
        for i, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            outs = net(images)
            loss = criterion(outs, labels)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()
            if i % print_interval == 0 and i > 0:
                print('Epoch: %d/%d  Iter: %d/%d  Loss: %.3f' % \
                    (epoch + 1, total_epochs, i, total_iters, avg_loss / print_interval))
                avg_loss = 0
    weights = 'cnn.pth'
    torch.save(net.state_dict(), weights)
    print('Weights are saved to: %s' % weights)

