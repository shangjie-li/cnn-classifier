import torchvision
import matplotlib.pyplot as plt
import numpy as np

from dataset import batch_size
from dataset import train_loader, test_loader
from dataset import CIFAR10_CLASSES


def show(img):
    img = img / 2 + 0.5
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    data_iter = iter(train_loader)
    images, labels = data_iter.next()
    print('Labels: ' + ' '.join('%5s' % CIFAR10_CLASSES[labels[j]] for j in range(batch_size)))
    show(torchvision.utils.make_grid(images))
