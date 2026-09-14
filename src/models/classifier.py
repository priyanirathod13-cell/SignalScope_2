import torch
import torch.nn as nn
import torchvision.models as models

class SignalScopeClassifier(nn.Module):
    """
    SignalScope Binary Transfer Learning Classifier
    Backbone: EfficientNet-B0 pretrained on ImageNet
    Classes: 0 = REAL, 1 = SYNTHETIC
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_features=True, unfreeze_top_block=True, dropout_rate=0.3):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Freeze lower layers if requested
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
            
            # Unfreeze the last convolutional projection block (features[8]) for fine-tuning
            if unfreeze_top_block:
                for param in self.backbone.features[8].parameters():
                    param.requires_grad = True

        # In EfficientNet-B0, features output has 1280 channels
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

def build_model(cfg):
    model_cfg = cfg.get("model", {})
    return SignalScopeClassifier(
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        freeze_features=model_cfg.get("freeze_features", True),
        unfreeze_top_block=model_cfg.get("unfreeze_top_block", True),
        dropout_rate=model_cfg.get("dropout_rate", 0.3)
    )
