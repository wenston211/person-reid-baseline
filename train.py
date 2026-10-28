"""
Training script for Person ReID
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import ReIDModel, TripletLoss, CrossEntropyLabelSmooth
from utils import RandomErasing, WarmupMultiStepLR, evaluate


def get_data(args):
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(10),
        transforms.RandomCrop((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets (assuming Market1501 format)
    train_dir = os.path.join(args.data_dir, 'bounding_box_train')
    query_dir = os.path.join(args.data_dir, 'query')
    gallery_dir = os.path.join(args.data_dir, 'bounding_box_test')

    train_set = ImageFolder(train_dir, transform=train_transform)
    query_set = ImageFolder(query_dir, transform=test_transform)
    gallery_set = ImageFolder(gallery_dir, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    query_loader = DataLoader(query_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    gallery_loader = DataLoader(gallery_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    num_classes = len(train_set.classes)
    return train_loader, query_loader, gallery_loader, num_classes


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_loader, query_loader, gallery_loader, num_classes = get_data(args)
    print(f"Number of classes: {num_classes}")

    # Model
    model = ReIDModel(num_classes=num_classes)
    model = model.to(device)

    # Loss functions
    ce_loss = CrossEntropyLabelSmooth(num_classes=num_classes)
    triplet_loss = TripletLoss(margin=0.3)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    # Scheduler with warmup
    scheduler = WarmupMultiStepLR(optimizer, milestones=[40, 70], gamma=0.1, warmup_epochs=10)

    # Training loop
    best_mAP = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, pids) in enumerate(train_loader):
            imgs = imgs.to(device)
            pids = pids.to(device)

            optimizer.zero_grad()
            cls_score, global_feat = model(imgs)

            loss_ce = ce_loss(cls_score, pids)
            loss_tri = triplet_loss(global_feat, pids)
            loss = loss_ce + loss_tri

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            cmc, mAP = evaluate(model, query_loader, gallery_loader, device)
            print(f"Evaluation - mAP: {mAP:.4f}, Rank-1: {cmc[0]:.4f}")

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
                print(f"New best mAP: {best_mAP:.4f}, model saved!")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print(f"Training complete. Best mAP: {best_mAP:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Person ReID Training')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to Market1501 dataset')
    parser.add_argument('--output_dir', type=str, default='./logs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=120, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3.5e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    args = parser.parse_args()
    train(args)
