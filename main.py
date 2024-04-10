from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
# system packages
from model import VGG19
from visualization import plot_image


# supports MacOS mps and CUDA
def GPU_Device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def load_image(img_path, max_size=400, shape=None):    
    image = Image.open(img_path).convert('RGB')
    ori_size = image.copy().size
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([ transforms.Resize(size), transforms.ToTensor()])
    
    # fake batch dimension required to fit network's input dimensions
    image = in_transform(image).unsqueeze(0)
    return image, ori_size




# dot product of feature map pairs
def gram_matrix(tensor):
    a, b, c, d = tensor.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = tensor.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


    
# compute the MSE loss between the target image and the style image
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, style):
        G = gram_matrix(style)
        self.loss = F.mse_loss(G, self.target)
        return style
    
    
# compute the MSE loss between the target image and the content image  
class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, content):
        self.loss = F.mse_loss(content, self.target)
        return content
    



# create a module to normalize input image so we can easily put it in a
# ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        mean = torch.tensor([0.485, 0.456, 0.406]).to(GPU_Device())
        std = torch.tensor([0.229, 0.224, 0.225]).to(GPU_Device())
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std
    
    


def get_style_model_and_losses(cnn, style_img, content_img,
                               content_layers=['conv_4'],
                               style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    # normalization module
    normalization = Normalization()
    # just in order to have an iterable access to or list of content/style
    # losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)
    
    
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)
    
    
    
    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses




def main():
    content_path = '../Datasets/flickr30k/flickr30k_images/81641.jpg'
    #content_path = '../Datasets/flickr30k/flickr30k_images/36979.jpg'
    style_path = '../Datasets/Best Artworks of All Time/images/Paul_Klee/Paul_Klee_107.jpg'
    #style_path = '../Datasets/Best Artworks of All Time/images/Paul_Klee/Paul_Klee_106.jpg'
    
    # load in content and style image
    content_img,_ = load_image(content_path)
    # Resize style to match content, makes code easier
    style_img,_ = load_image(style_path, shape=content_img.shape[-2:])
    
    # move the data to GPU
    content_img = content_img.to(GPU_Device())
    style_img = style_img.to(GPU_Device())
    
    plot_image(content_img, style_img)
    
    cnn = VGG19().get_feature_layers()
    
    cnn = cnn.to(GPU_Device())
    
    # target
    input_img = content_img.clone()
    
    # build the cnn model
    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

    num_steps=300
    style_weight=1000000
    content_weight=1
                           
                           
    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    iteration = [0] # pass by reference
    while iteration[0] <= num_steps:
        
        def closure():
            # clip the image to [0,1]
            with torch.no_grad():
                input_img.clamp_(0, 1)
        
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
        
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
        
            style_score *= style_weight
            content_score *= content_weight
        
            loss = style_score + content_score
            loss.backward()
            
            iteration[0] += 1
            if iteration[0] % 50 == 0:
                print(f"Iteration {iteration}:")
                print(f"Style Loss: {style_score:.4f} Content Loss: {content_score:.4f}")
                print()
                plot_image(content_img, style_img, input_img.clone().detach().clip(0, 1))
            
            return style_score + content_score

        optimizer.step(closure)
        

    
if __name__ == "__main__":
    main()