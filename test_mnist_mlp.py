import numpy as np
import argparse
import torch
import torchvision
from PIL import Image

from net import MLP

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='PyTorch example: MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--image', '-i', type=str, default="./result/mnist/sample.png",
                        help='pass to input image')
    parser.add_argument('--resume_path', '-r', default='./result/mnist/model.ckpt',
                        help='path to the training model')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()
    model = MLP(n_units=args.unit, n_in=784, n_out=10)
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
    
    transform = torchvision.transforms.ToTensor()
    
    img_pil = Image.open(args.image).convert("L").resize((28,28))
    img = transform(img_pil).reshape(-1, 28*28)
    img = img.to(device)

    result = model(img)
    _, predicted = torch.max(result.data, 1)
    print("predict:", predicted.item())
    
    
if __name__ == '__main__':
    main()
