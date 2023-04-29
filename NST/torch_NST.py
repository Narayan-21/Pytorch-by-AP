# Neural Style Transfer(NST) technique implementation.

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights, vgg19
from torchvision.utils import save_image

model = models.vgg19(weights=VGG19_Weights.DEFAULT).features


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.choosen_features = ['0','5','10','19','28']
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT).features[:29] # Getting only the layers that will be used in the loss function.
    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.choosen_features:
                features.append(x)
        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image

# device = torch.device("cuda" if torch.cuda.is_available else "cpu")

image_size = 356

loader = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[], std=[])
    ]
)

original_image = load_image("NST/musk.jpg")
style_image = load_image("NST/style_1.jpg")

model = VGG().eval()

# generated = torch.randn(original_image.shape, requires_grad=True) # taking a random image as our initial generated pic.
generated = original_image.clone().requires_grad_(True)

# Hyperparameters
total_steps = 6000
learning_rate = 0.01
alpha = 1
beta = 0.01

optimizer = optim.Adam([generated], lr=learning_rate)

for step in range(total_steps):
    generated_features = model(generated)
    original_image_features = model(original_image)
    style_features = model(style_image)

    style_loss = content_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_image_features, style_features
    ):
        batch_size, channel, height, width = gen_feature.shape
        content_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # Compute Gram matrix
        G = gen_feature.view(channel, height*width).mm(gen_feature.view(channel, height*width).t())
        A = style_feature.view(channel, height*width).mm(style_feature.view(channel, height*width).t())
        style_loss += torch.mean((G - A)**2)

    total_loss = alpha * content_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(total_loss)
        save_image(generated, 'NST/generated_1.jpg')