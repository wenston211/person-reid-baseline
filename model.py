"""
Person ReID Model based on ResNet50-IBN + BNNeck + BoT tricks
"""
import torch
import torch.nn as nn
import torchvision.models as models


class IBN(nn.Module):
    """Instance-Batch Normalization layer"""
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        half2 = planes - half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(half2)

    def forward(self, x):
        split = torch.split(x, int(x.size(1) / 2), 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class ResNet50IBN(nn.Module):
    """ResNet50 with IBN-a blocks"""
    def __init__(self, pretrained=True):
        super(ResNet50IBN, self).__init__()
        resnet = models.resnet50(pretrained=pretrained)

        # Replace BN in layer1 and layer2 with IBN
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Insert IBN after layer1 and layer2 first block
        self._replace_with_ibn()

        self.gap = nn.AdaptiveAvgPool2d(1)

    def _replace_with_ibn(self):
        # Simple IBN replacement (simplified version)
        for layer in [self.layer1, self.layer2]:
            for block in layer.children():
                if hasattr(block, 'bn3'):
                    block.bn3 = IBN(block.bn3.num_features)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class ReIDModel(nn.Module):
    """
    Person ReID Model with:
    - Backbone: ResNet50-IBN
    - BNNeck
    - Classifier head
    """
    def __init__(self, num_classes=751, feature_dim=2048):
        super(ReIDModel, self).__init__()
        self.backbone = ResNet50IBN(pretrained=True)

        # BNNeck
        self.bottleneck = nn.BatchNorm1d(feature_dim)
        self.bottleneck.bias.requires_grad_(False)  # no bias

        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

    def forward(self, x, return_features=False):
        # Backbone
        global_feat = self.backbone(x)  # (batch, 2048)

        # BNNeck
        feat = self.bottleneck(global_feat)  # for inference

        if self.training:
            # Training: return both global feat (for triplet) and bn feat (for ce)
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feat for triplet loss
        else:
            # Inference: return normalized feature
            if return_features:
                return feat
            # L2 normalize
            feat = nn.functional.normalize(feat, dim=1)
            return feat


class TripletLoss(nn.Module):
    """Triplet loss with hard mining"""
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        # Compute pairwise distance
        n = feats.size(0)
        dist = torch.pow(feats, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(feats, feats.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # numerical stability

        # For each anchor, find hardest positive and hardest negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing"""
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
