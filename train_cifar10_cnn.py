import argparse

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from net import Cifar_CNN
import torch
import torch.nn as nn
import torchvision

def main():
    parser = argparse.ArgumentParser(description='PyTorch example: Cifar10 with CNN')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result/cifar10/',
                        help='Directory to output the result')
    parser.add_argument('--resume_path', '-r', default='',
                        help='Resume the training from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')
    os.makedirs(args.out, exist_ok=True)

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = Cifar_CNN(n_in=3, n_out=10)
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    model = model.to(device)  # Copy the model to the GPU
    
    if args.resume_path != '':
        pretrained_weights = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(pretrained_weights)

    # Setup an optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    

    ### Define dataset
    train_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10',
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10',
                                          train=False, 
                                          transform=torchvision.transforms.ToTensor())
    ### Define dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batchsize, 
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batchsize, 
                                            shuffle=False)

    # Define objective
    criterion = nn.CrossEntropyLoss()

    accuracies = {
        'train': [],
        'test': []    
    }
    
    total_step = len(train_loader)
    for epoch in range(args.epoch):
        
        model.train()
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device) # DIfferent from mlp version, because conv needs channel, height, width axis
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Caliculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i+1) % 200 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, args.epoch, i+1, total_step, loss.item()))
        accuracy_train = 100 * correct / total
        print('Train Accuracy: {}'.format(accuracy_train))
        accuracies['train'].append(accuracy_train)
        
        model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device) # DIfferent from mlp version, because conv needs channel, height, width axis
            labels = labels.to(device)
            with torch.no_grad():
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy_test = 100 * correct / total
        print('Test Accuracy: {}'.format(accuracy_test))
        accuracies['test'].append(accuracy_test)

        # Save weights
        torch.save(model.state_dict(), os.path.join(args.out, 'model_{}.ckpt'.format(epoch)))


    # Plot
    plt.ylim([40,100])
    plt.xlim([0,args.epoch+1])
    x = range(1, args.epoch+1)
    plt.plot(x, accuracies['train'], label='train', marker='x')
    plt.plot(x, accuracies['test'], label='test', marker='x')
    plt.legend()
    plt.savefig(os.path.join(args.out, 'accuracy.jpg'))

    
if __name__ == '__main__':
    main()