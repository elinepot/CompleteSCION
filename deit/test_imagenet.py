import torchvision

path = '/work/fra/shared/bases_dsam/ImageNet/'
trainset = torchvision.datasets.ImageNet(root=path, split='val')