import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # Placeholders
        self.activations = None
        self.gradients = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, rgb, noise, freq, ela,class_idx=None):

        self.model.zero_grad()

        logit, _ = self.model(rgb, noise, freq, ela)
        confidence = torch.sigmoid(logit)
        if class_idx is None:
            class_idx = (confidence >= 0.5).float()  
            class_idx = class_idx.view_as(logit)  
        # Take positive or negative class depending on prediction
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, class_idx
            )
        loss.backward(retain_graph=True)

        # Global Average Pooling on gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B,C,1,1]
        cam = (weights * self.activations).sum(dim=1)            # [B,H,W]

        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return confidence,cam



def get_gradcam(model, batch, device, filename="gradcam.png" , save_dir="uploads/gradcam"):
    model.eval()
    
    # Use only one sample
    rgb = batch['rgb'].to(device).unsqueeze(0)
    noise = batch['noise'].to(device).unsqueeze(0)  
    freq = batch['freq'].to(device).unsqueeze(0)
    ela = batch['ela'].to(device).unsqueeze(0)
  
    target_layer =  model.rgb_enc.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    confidence,cam = gradcam.generate(rgb, noise, freq, ela)  # [1,H,W]
    print(confidence)
    cam = cam.squeeze().cpu().numpy()

    img = rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    # Resize CAM to image resolution
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.5 * img + 0.5 * heatmap

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save image
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, (overlay * 255).astype(np.uint8))

    return confidence,save_path

def visualize_gradcam(model, batch, device):
    model.eval()

    # Use only one sample
    rgb = batch['rgb'][0:1].to(device)
    noise = batch['noise'][0:1].to(device)
    freq = batch['freq'][0:1].to(device)
    ela = batch['ela'][0:1].to(device)
    y = batch['label'][0:1].float().to(device)
    # Choose last conv layer of ResNet18
    target_layer = model.rgb_enc.backbone.layer4[-1]

    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(rgb, noise, freq, ela,y=y)  # [1,H,W]
    cam = cam.squeeze().cpu().numpy()
    img = rgb.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    overlay = 0.5 * img + 0.5 * heatmap


    # Plot
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")

    plt.show()

