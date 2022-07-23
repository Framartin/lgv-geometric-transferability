import torch
from PIL import Image
from torchvision.transforms import functional as F


class RandomResizePad(torch.nn.Module):
    def __init__(self, min_resize):
        super().__init__()
        self.min_resize = min_resize

    def forward(self, img):
        size_original = img.size()
        if size_original[-1] != size_original[-2]:
            raise ValueError("Only squared images supported")
        random_size = int(torch.randint(low=self.min_resize, high=size_original[-1], size=(1,)))
        pad_margin = size_original[-1] - random_size
        img = F.resize(img, size=random_size, interpolation=Image.NEAREST)
        pad_top = int(torch.randint(low=0, high=pad_margin, size=(1,)))
        pad_bottom = pad_margin - pad_top
        pad_right = int(torch.randint(low=0, high=pad_margin, size=(1,)))
        pad_left = pad_margin - pad_right
        img = F.pad(img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0)
        if img.size() != size_original:
            raise RuntimeError("Output size is not the same than input size.")
        return img
