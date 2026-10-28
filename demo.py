"""
Inference demo for Person ReID
"""
import os
import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import ReIDModel


def extract_feature(model, image, transform, device):
    """Extract feature from a single image"""
    model.eval()
    with torch.no_grad():
        img = transform(image).unsqueeze(0).to(device)
        feat = model(img)
    return feat.cpu()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load model
    model = ReIDModel(num_classes=args.num_classes)
    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Extract query feature
    query_img = Image.open(args.query_image).convert('RGB')
    query_feat = extract_feature(model, query_img, transform, device)

    # Extract gallery features
    gallery_feats = []
    gallery_names = []
    for img_name in sorted(os.listdir(args.gallery_dir)):
        if img_name.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(args.gallery_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            feat = extract_feature(model, img, transform, device)
            gallery_feats.append(feat)
            gallery_names.append(img_name)

    gallery_feats = torch.cat(gallery_feats, dim=0)

    # Compute cosine similarity
    query_feat = torch.nn.functional.normalize(query_feat, dim=1)
    gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1)
    similarity = torch.mm(query_feat, gallery_feats.t()).squeeze(0)

    # Sort by similarity
    scores, indices = similarity.sort(descending=True)

    print(f"\nQuery image: {args.query_image}")
    print(f"\nTop {args.top_k} matches:")
    for i in range(min(args.top_k, len(gallery_names))):
        print(f"  {i+1}. {gallery_names[indices[i]]} - similarity: {scores[i]:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Person ReID Demo')
    parser.add_argument('--query_image', type=str, required=True, help='Path to query image')
    parser.add_argument('--gallery_dir', type=str, required=True, help='Path to gallery directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--num_classes', type=int, default=751, help='Number of classes (Market1501: 751)')
    parser.add_argument('--top_k', type=int, default=10, help='Top K matches to show')
    args = parser.parse_args()
    main(args)
