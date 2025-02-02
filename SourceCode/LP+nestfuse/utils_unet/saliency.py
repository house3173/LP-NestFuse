import logging
import warnings
from pathlib import Path

import cv2
import torch.hub
from kornia import image_to_tensor, tensor_to_image
from torchvision.transforms import Resize, Compose, Normalize

from utils_unet.u2net import U2NETP, U2NET

class Saliency:
    r"""
    Init saliency detection pipeline to generate mask from infrared images.
    """

    def __init__(self):
        # init device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'deploy u2net on device {str(device)}')
        self.device = device

        # init u2net small (u2netp)
        net = U2NETP(in_ch=1, out_ch=1)
        self.net = net

        # download pretrained parameters
        # ckpt_p = Path.cwd() / 'utils' / 'u2netp.pth'
        ckpt_p = "C:/PVH/SourceCode/LP+nestfuse/utils_unet/u2netp.pth"
        ckpt = torch.load(ckpt_p, map_location=device)
        net.load_state_dict(ckpt)

        # move to device
        net.to(device)

        # more parameters
        self.transform_fn = Compose([Resize(size=(320, 320)), Normalize(mean=0.485, std=0.229)])

    @torch.inference_mode()
    def inference(self, img_path: str | Path):
        img_path = Path(img_path)

        # Read and preprocess image
        img = self._imread(img_path).to(self.device)
        original_size = img.shape[-2:]  # Store original size for resizing later
        img = self.transform_fn(img)
        img = img.unsqueeze(0)  # Add batch dimension

        # Perform inference
        self.net.eval()
        mask = self.net(img)[0]

        # Normalize and resize mask back to original size
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        reverse_fn = Resize(size=original_size)
        mask = reverse_fn(mask).squeeze()

        # Convert mask to numpy format and return
        return tensor_to_image(mask) * 255

    @staticmethod
    def _imread(img_p: str | Path):
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        img = image_to_tensor(img).float() / 255
        return img
