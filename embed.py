"""
embed.py
- MobileNetEmbed class for fast embeddings
- optional CLIP extractor if GPU available
- simclr_refine for per-video self-supervision (very small)
"""

import torch, torch.nn as nn, numpy as np
import torchvision, torchvision.transforms as T

class MobileNetEmbed(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        m = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT)
        self.backbone = nn.Sequential(*(list(m.features.children())))
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.to(device)
    def forward(self, x):
        with torch.no_grad():
            f = self.backbone(x)
            f = self.pool(f).flatten(1)
        return f

_transform = T.Compose([T.ToTensor(),
                       T.ConvertImageDtype(torch.float32),
                       T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def batch_embed(model, imgs, batch=64, device='cpu'):
    embs = []
    for i in range(0, len(imgs), batch):
        x = torch.stack([_transform(img) for img in imgs[i:i+batch]]).to(device)
        with torch.no_grad():
            e = model(x).cpu().numpy()
        embs.append(e)
    return np.vstack(embs)

# Simple SimCLR-lite refinement (per-video)
class TinyProj(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, out_dim))
    def forward(self, x): return self.net(x)

def simclr_refine(embs, epochs=3, batch=64, lr=1e-3, temp=0.1, device='cpu'):
    X = torch.from_numpy(embs).float().to(device)
    proj = TinyProj(embs.shape[1]).to(device)
    opt = torch.optim.Adam(proj.parameters(), lr=lr)
    for ep in range(epochs):
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), batch):
            idx = perm[i:i+batch]
            x1 = X[idx]
            x2 = x1 + 0.01*torch.randn_like(x1)
            z1 = nn.functional.normalize(proj(x1), dim=1)
            z2 = nn.functional.normalize(proj(x2), dim=1)
            sim = (z1 @ z2.T)/temp
            labels = torch.arange(sim.size(0)).to(device)
            loss = nn.CrossEntropyLoss()(sim, labels)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = nn.functional.normalize(proj(torch.from_numpy(embs).float().to(device)), dim=1).cpu().numpy()
    return z

# Optional CLIP extractor (if GPU and openAI/clip installed)
def clip_embeddings(frames, model=None, processor=None, device='cpu'):
    try:
        from transformers import CLIPProcessor, CLIPModel
    except Exception as e:
        raise ImportError("Install transformers for CLIP")
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    imgs = [__pil_from_np(img) for img in frames]
    inputs = processor(images=imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        embs = model.get_image_features(**inputs)
    return embs.cpu().numpy()

def __pil_from_np(arr):
    from PIL import Image
    return Image.fromarray(arr.astype('uint8'), 'RGB')
