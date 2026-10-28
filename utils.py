"""
Utility functions for Person ReID
"""
import torch
import numpy as np
from PIL import Image


class RandomErasing(object):
    """Randomly erase a rectangle region in an image (Zhong et al., AAAI 2020)"""
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if np.random.uniform() > self.probability:
            return img

        for _ in range(100):
            area = img.size()[1] * img.size()[2]
            target_area = np.random.uniform(self.sl, self.sh) * area
            aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = np.random.randint(0, img.size()[1] - h)
                y1 = np.random.randint(0, img.size()[2] - w)
                for i in range(img.size()[0]):
                    img[i, x1:x1 + h, y1:y1 + w] = self.mean[i]
                return img
        return img


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler with warmup"""
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_epochs=10, warmup_factor=1.0 / 3):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super().__init__(optimizer)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        lr = []
        for base_lr in self.base_lrs:
            for milestone in self.milestones:
                if self.last_epoch >= milestone:
                    base_lr *= self.gamma
            lr.append(base_lr * warmup_factor)
        return lr


def evaluate(model, query_loader, gallery_loader, device):
    """Evaluate ReID model: compute CMC and mAP"""
    model.eval()

    def extract(loader):
        feats, pids, camids = [], [], []
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(device)
                output = model(imgs)
                feats.append(output.cpu())
                pids.append(targets)
        feats = torch.cat(feats, dim=0)
        pids = torch.cat(pids, dim=0)
        return feats, pids

    q_feats, q_pids = extract(query_loader)
    g_feats, g_pids = extract(gallery_loader)

    # Compute distance matrix
    q_feats = torch.nn.functional.normalize(q_feats, dim=1)
    g_feats = torch.nn.functional.normalize(g_feats, dim=1)
    dist_mat = 1 - torch.mm(q_feats, g_feats.t())

    dist_mat = dist_mat.numpy()
    q_pids = q_pids.numpy()
    g_pids = g_pids.numpy()

    # Compute CMC and mAP
    num_q = q_pids.shape[0]
    cmc = np.zeros(g_pids.shape[0])
    all_AP = []

    for i in range(num_q):
        order = np.argsort(dist_mat[i])
        matches = (g_pids[order] == q_pids[i]).astype(np.int32)

        # CMC
        raw_cmc = matches.cumsum()
        cmc += (raw_cmc > 0).astype(np.float32)

        # AP
        n_rel = matches.sum()
        if n_rel > 0:
            recall = raw_cmc / n_rel
            precision = raw_cmc / (np.arange(len(matches)) + 1)
            AP = (precision * matches).sum() / n_rel
            all_AP.append(AP)

    cmc = cmc / num_q
    mAP = np.mean(all_AP) if len(all_AP) > 0 else 0.0

    return cmc, mAP
