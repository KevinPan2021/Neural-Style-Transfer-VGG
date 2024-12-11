import torch
import torch.nn as nn
import sys

path = '../pretrained_models/VGG/'
sys.path.append(path)

from torchvision.models import vgg19


class VGG19(nn.Module):
    def __init__(self,):
        super(VGG19, self).__init__()
        
        self.model = vgg19(weights=None)
        
        weights = torch.load(path + 'vgg19-IMAGENET1K_V1.pth')
        self.model.load_state_dict(weights)
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    
    # the feature extraction from model
    def get_feature_layers(self,):
        return self.model.features
    
        
    def forward(self, x):
        return self.model(x)
    



def main():
    model = VGG19()
    
    
if __name__ == "__main__":
    main()
    