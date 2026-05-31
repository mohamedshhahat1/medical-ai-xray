"""
Grad-CAM Visualization for X-Ray Model Interpretability
=========================================================

Generates heatmaps showing which regions of the X-ray the model
focuses on when making a prediction. Critical for medical AI
where explainability is required.

Grad-CAM (Gradient-weighted Class Activation Mapping):
    Uses the gradients flowing into the final conv layer to produce
    a coarse localization map highlighting important regions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io


class GradCAM:
    """
    Grad-CAM visualization for CNN models.

    Usage:
        grad_cam = GradCAM(model, target_layer='layer4')
        heatmap = grad_cam.generate(input_tensor, class_idx=1)
        overlay = grad_cam.overlay_on_image(original_image, heatmap)
    """

    def __init__(self, model, target_layer_name="layer4"):
        """
        Args:
            model (nn.Module): The classification model.
            target_layer_name (str): Name of the conv layer to visualize.
                For ResNet: 'layer4' (last conv block)
                For DenseNet: 'features.denseblock4'
        """
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        target_layer = dict(model.named_modules()).get(target_layer_name)
        if target_layer is None:
            # Fallback: try to find the last conv layer
            for name, module in reversed(list(model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    break

        if target_layer:
            target_layer.register_forward_hook(self._forward_hook)
            target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor (torch.Tensor): Preprocessed input (1, 3, H, W).
            class_idx (int, optional): Target class. If None, uses predicted class.

        Returns:
            np.ndarray: Heatmap of shape (H, W) with values in [0, 1].
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for the target class
        target = output[0, class_idx]
        target.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros((7, 7))

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted sum of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    def overlay_on_image(self, original_image, heatmap, alpha=0.4):
        """
        Overlay the heatmap on the original image.

        Args:
            original_image (PIL.Image): Original X-ray image.
            heatmap (np.ndarray): Grad-CAM heatmap.
            alpha (float): Transparency of the overlay.

        Returns:
            PIL.Image: Image with heatmap overlay.
        """
        # Resize heatmap to image size
        img_array = np.array(original_image.convert("RGB"))
        h, w = img_array.shape[:2]

        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
        ).astype(np.float32) / 255.0

        # Create colormap (blue → green → red)
        colormap = np.zeros((h, w, 3), dtype=np.float32)
        colormap[:, :, 0] = heatmap_resized  # Red channel
        colormap[:, :, 1] = heatmap_resized * 0.5  # Green (partial)
        colormap[:, :, 2] = 1.0 - heatmap_resized  # Blue (inverse)

        # Blend
        overlay = (1 - alpha) * img_array.astype(np.float32) / 255.0 + alpha * colormap
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(overlay)
