import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.models as models
import torchvision
import torchvision.transforms as transforms

class MergeAutoencoder(nn.Module):
    def __init__(self, colorization_resnet, jigsaw_resnet, rotation_resnet, simsiam_resnet):
        super(MergeAutoencoder, self).__init__()
        # feature extractors
        self.colorization_resnet = colorization_resnet
        self.jigsaw_resnet = jigsaw_resnet
        self.rotation_resnet = rotation_resnet
        self.simsiam_resnet = simsiam_resnet
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(3584, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1536),
            nn.ReLU(True),
            nn.Linear(1536, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Linear(512, 256), 
            nn.Linear(256, 10))
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1536),
            nn.ReLU(True),
            nn.Linear(1536, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 3584),
            nn.Tanh())

    def forward(self, x):
        jigsaw = self.jigsaw_resnet(x)
        rotation = self.rotation_resnet(x)
        simsiam = self.simsiam_resnet(x)
        gray_x = transforms.Grayscale()(x)
        color = self.colorization_resnet(gray_x)
        color = color.view(jigsaw.shape[0], -1)
        # print(jigsaw.size())
        # print(rotation.size())
        # print(simsiam.size())
        # print(color.size())
        
        feature_map = torch.cat((color, jigsaw, rotation, simsiam), dim=1)
        # print(feature_map.size())
        latent_vec = self.encoder(feature_map)
        # out = self.decoder(latent_vec)
        return latent_vec