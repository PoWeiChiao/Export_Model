import torch
import torchvision.models as models

def main():
    resnet = torch.jit.trace(models.resnet18(pretrained=True), torch.rand(1, 3, 224, 224))
    output = resnet(torch.ones(1, 3, 224, 224))
    print(output)
    resnet.save('resnet.pt')

if __name__ == '__main__':
    main()