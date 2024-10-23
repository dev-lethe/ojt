import argparse


import torch
import torchvision
from net import Cifar_CNN

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Practice: Cifar10')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume_path', '-r', default='result/cifar10/model_19.ckpt',
                        help='Path to the model')
    parser.add_argument('--dataset', '-d', default='mini_cifar/test',
                        help='Directory for train mini_cifar')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('')

    model = Cifar_CNN(n_in=3, n_out=10)

    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    
    if args.resume_path != '':
        print("Loaded pretrained weights")
        pretrained_weights = torch.load(args.resume_path, map_location=device)
        model.load_state_dict(pretrained_weights)
        
    model = model.to(device)  # Copy the model to the GPU
    model.eval()

    # Load the Cifar-10 test set
    test_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10',
                                          train=False, 
                                          transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=args.batchsize, 
                                            shuffle=False)
    print('num of test set images : {}'.format(len(test_dataset)))

    
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


if __name__ == '__main__':
    main()