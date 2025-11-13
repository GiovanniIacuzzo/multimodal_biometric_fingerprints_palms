from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

class FingerprintSegmentationModel(nn.Module):
    def __init__(self, num_labels=1, image_size=(512,512), pretrained_model="nvidia/segformer-b2-finetuned-ade-512-512"):
        super().__init__()
        self.num_labels = num_labels
        self.image_size = image_size
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x).logits
        outputs = nn.functional.interpolate(outputs, size=self.image_size, mode="bilinear", align_corners=False)
        return outputs
