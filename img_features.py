import numpy as np

import torch
from torchvision import models, transforms

def resnet50_features(imgs):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(weights="IMAGENET1K_V2").to(device)
    layer = model._modules.get('avgpool')
    model.eval()

    t = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    imgs = torch.stack([t(img) for img in imgs]).to(device)
    embedds = torch.zeros((len(imgs), 2048, 1, 1)).to(device)

    def copy_data(m, i, o):
        embedds.copy_(o.data)

    h = layer.register_forward_hook(copy_data)
    with torch.no_grad():
        model(imgs)
    h.remove()
    return embedds.squeeze(dim=3).squeeze(dim=2).cpu().numpy()
