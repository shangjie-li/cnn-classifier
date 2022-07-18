import torch

from dataset import train_loader, test_loader
from dataset import CIFAR10_CLASSES
from cnn import CNN


if __name__ == '__main__':
    net = CNN()
    weights = 'cnn.pth'
    print('Loading weights from %s...' % weights)
    state_dict = torch.load(weights)
    net.load_state_dict(state_dict)
    
    print('Starting evaluating...')
    correct_preds = {classname: 0 for classname in CIFAR10_CLASSES}
    total_gts = {classname: 0 for classname in CIFAR10_CLASSES}
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outs = net(images)
            _, preds = torch.max(outs, 1)
            for label, pred in zip(labels, preds):
                if label == pred:
                    correct_preds[CIFAR10_CLASSES[label]] += 1
                total_gts[CIFAR10_CLASSES[label]] += 1
    avg_acc = 0
    print('-------------------------------------')
    for classname, gt_num in total_gts.items():
        acc = correct_preds[classname] / gt_num
        avg_acc += acc
        print('Accuracy for class %s is: %.3f' % (classname, acc))
    print('-------------------------------------')
    print('Mean Accuracy of the model is: %.3f' % (avg_acc / len(CIFAR10_CLASSES)))
