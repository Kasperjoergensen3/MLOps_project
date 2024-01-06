from torch import nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, BeitForImageClassification
import torch
from datasets import load_dataset

class BEiT(nn.Module):
    def __init__(self, image_processor, model):
        super(BEiT, self).__init__()
        self.image_processor = image_processor
        self.model = model

    def forward(self, image):
        inputs = self.image_processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        x = F.log_softmax(logits, dim=1)
        return x


def load_pretrained(filepath):
    base_patch = "microsoft/beit-base-patch16-224"
    image_processor = AutoImageProcessor.from_pretrained(base_patch)
    model = BeitForImageClassification.from_pretrained(base_patch)
    my_model = BEiT(image_processor, model)
    return my_model
