import torch
import torchvision

from dataset import batch_size
from dataset import train_loader, test_loader
from dataset import CIFAR10_CLASSES
from cnn import CNN
from dataset_player import show


if __name__ == '__main__':
    net = CNN()
    weights = 'cnn.pth'
    print('Loading weights from %s...' % weights)
    state_dict = torch.load(weights)
    net.load_state_dict(state_dict)

    data_iter = iter(test_loader)
    images, labels = data_iter.next()
    with torch.no_grad():
        outs = net(images)
        _, preds = torch.max(outs, 1)
    print('Predictions: ' + ' '.join('%5s' % CIFAR10_CLASSES[preds[j]] for j in range(batch_size)))
    show(torchvision.utils.make_grid(images))
