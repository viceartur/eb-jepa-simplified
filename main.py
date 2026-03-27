import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.transforms.functional as TF
from torchvision.utils import draw_bounding_boxes, make_grid
from tqdm import tqdm
import wandb


class RandomResizedCrop:
    """Random resized crop augmentation."""

    def __init__(self, size, scale=(0.2, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img):
        return transforms.RandomResizedCrop(self.size, scale=self.scale)(img)


class ColorJitter:
    """Color jitter augmentation."""

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8):
        self.transform = transforms.ColorJitter(brightness, contrast, saturation, hue)
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return self.transform(img)
        return img


class Grayscale:
    """Grayscale augmentation."""

    def __init__(self, prob=0.2):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.Grayscale(num_output_channels=3)(img)
        return img


class HorizontalFlip:
    """Horizontal flip augmentation."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if torch.rand(1) < self.prob:
            return transforms.functional.hflip(img)
        return img


def get_train_transforms():
    """Get training transforms for self-supervised learning."""
    transform = transforms.Compose(
        [
            RandomResizedCrop(32, scale=(0.2, 1.0)),
            ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, prob=0.8
            ),
            Grayscale(prob=0.2),
            HorizontalFlip(prob=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )

    return transform


def get_val_transforms():
    """Get validation transforms."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )


class ImageDataset(torch.utils.data.Dataset):
    """Custom dataset that applies augmentations multiple times to create views."""

    def __init__(self, dataset, transform, num_crops=2):
        self.dataset = dataset
        self.transform = transform
        self.num_crops = num_crops

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        views = [self.transform(image) for _ in range(self.num_crops)]
        return views, label


class BCS(nn.Module):
    """BCS (Batched Characteristic Slicing) loss for SIGReg."""

    def __init__(self, num_slices, lmbd):
        super().__init__()
        self.num_slices = num_slices
        self.step = 0
        self.lmbd = lmbd

    def epps_pulley(self, x, t_min=-3, t_max=3, n_points=10):
        """Epps-Pulley test statistic for Gaussianity."""
        # integration points
        t = torch.linspace(t_min, t_max, n_points, device=x.device)
        # theoretical CF for N(0, 1)
        exp_f = torch.exp(-0.5 * t**2)
        # ECF
        x_t = x.unsqueeze(2) * t  # (N, M, T)
        ecf = (1j * x_t).exp().mean(0)
        # weighted L2 distance
        err = exp_f * (ecf - exp_f).abs() ** 2
        T = torch.trapz(err, t, dim=1)
        return T

    def forward(self, z1, z2):
        with torch.no_grad():
            dev = z1.device
            g = torch.Generator(device=dev)
            g.manual_seed(self.step)
            proj_shape = (z1.size(1), self.num_slices)
            A = torch.randn(proj_shape, device=dev, generator=g)
            A /= A.norm(p=2, dim=0)
        view1 = z1 @ A
        view2 = z2 @ A

        self.step += 1
        bcs = (self.epps_pulley(view1).mean() + self.epps_pulley(view2).mean()) / 2
        invariance_loss = F.mse_loss(z1, z2).mean()
        total_loss = invariance_loss + self.lmbd * bcs
        return {"loss": total_loss, "bcs_loss": bcs, "invariance_loss": invariance_loss}


class LinearProbe(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #output 64 channels

        self.l1_b1_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l1_b1_bn1 = nn.BatchNorm2d(64)
        self.l1_b1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l1_b1_bn2 = nn.BatchNorm2d(64)
        #output 64 channels

        self.l1_b2_conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l1_b2_bn1 = nn.BatchNorm2d(64)
        self.l1_b2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.l1_b2_bn2 = nn.BatchNorm2d(64)
        #output 64 channels

        self.l2_b1_shortcut_conv = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.l2_b1_shortcut_bn = nn.BatchNorm2d(128)
        #output 128 channels

        self.l2_b1_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.l2_b1_bn1 = nn.BatchNorm2d(128)
        self.l2_b1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l2_b1_bn2 = nn.BatchNorm2d(128)
        #output 128 channels

        self.l2_b2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l2_b2_bn1 = nn.BatchNorm2d(128)
        self.l2_b2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.l2_b2_bn2 = nn.BatchNorm2d(128)
        #output 128 channels

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #output 64 channels

        identity = x
        out = self.l1_b1_conv1(x)
        out = self.l1_b1_bn1(out)
        out = self.relu(out)
        out = self.l1_b1_conv2(out)
        out = self.l1_b1_bn2(out)
        x = self.relu(out + identity)
        #output 64 channels


        identity = x
        out = self.l1_b2_conv1(x)
        out = self.l1_b2_bn1(out)
        out = self.relu(out)
        out = self.l1_b2_conv2(out)
        out = self.l1_b2_bn2(out)
        x = self.relu(out + identity)
        #output 64 channels


        identity = self.l2_b1_shortcut_conv(x)
        identity = self.l2_b1_shortcut_bn(identity)
        #output 128 channels

        out = self.l2_b1_conv1(x)
        out = self.l2_b1_bn1(out)
        out = self.relu(out)
        out = self.l2_b1_conv2(out)
        out = self.l2_b1_bn2(out)
        x = self.relu(out + identity)
        #output 128 channels


        identity = x
        out = self.l2_b2_conv1(x)
        out = self.l2_b2_bn1(out)
        out = self.relu(out)
        out = self.l2_b2_conv2(out)
        out = self.l2_b2_bn2(out)
        x = self.relu(out + identity)
        #output 128 channels


        x = self.avgpool(x)
        x = x.squeeze()
        
        return x


def train_epoch(model, linear_probe, train_loader, optimizer, probe_optimizer, device, epoch, loss_fn):
    model.train()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    epoch_ssl_loss = 0.0
    epoch_linear_loss = 0.0

    for batch_idx, (views, target) in enumerate(pbar):
        view1, view2 = views[0].to(device), views[1].to(device)
        target = target.to(device)

        features1 = model(view1)
        features2 = model(view2)

        ssl_loss = loss_fn(features1, features2)["loss"]

        features_frozen = features1.detach().float()
        linear_outputs = linear_probe(features_frozen)
        linear_loss = F.cross_entropy(linear_outputs, target)

        total_loss = ssl_loss + linear_loss

        optimizer.zero_grad()
        probe_optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()
        probe_optimizer.step()

        epoch_ssl_loss += ssl_loss.item()
        epoch_linear_loss += linear_loss.item()

        pbar.set_postfix({
            "ssl_loss": f"{ssl_loss.item():.4f}",
            "linear_loss": f"{linear_loss.item():.4f}",
        })

    avg_ssl_loss = epoch_ssl_loss / len(train_loader)
    avg_linear_loss = epoch_linear_loss / len(train_loader)
    
    return avg_ssl_loss, avg_linear_loss, features1.detach()


def evaluate_linear_probe(model, linear_probe, val_loader, device):
    """Evaluate linear probe on validation set."""
    model.eval()
    linear_probe.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            features = model(data)

            outputs = linear_probe(features.float())
            loss = F.cross_entropy(outputs, target)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(val_loader)

    return accuracy, avg_loss


def views_visualization(base_dataset, num_images, filename):
    images_with_boxes = []

    for i in range(num_images):
        img, _ = base_dataset[i]

        params1 = transforms.RandomResizedCrop.get_params(img, scale=(0.2, 1.0), ratio=(3./4., 4./3.))
        params2 = transforms.RandomResizedCrop.get_params(img, scale=(0.2, 1.0), ratio=(3./4., 4./3.))
    
        img_tensor = TF.to_tensor(img).mul(255).byte()

        def get_bbox(p):
            top, left, h, w = p
            return [left, top, min(left+w, 32), min(top+h, 32)]
        
        boxes = torch.tensor([get_bbox(params1), get_bbox(params2)], dtype=torch.int)
        colors = ["yellow", "purple"]

        res_img = draw_bounding_boxes(img_tensor, boxes, colors=colors, width=1)
        images_with_boxes.append(res_img)

    grid = make_grid(images_with_boxes, nrow=5)
    grid_pil = TF.to_pil_image(grid)
    grid_pil.save(filename)

    print("views plot saved")


def main(args):
    use_wandb = args.wandb.lower() == 'true'

    if use_wandb:
        wandb.init(
            project="dl-finalproject-image-ssl-cifar10",
            config={
                "epochs": 300,
                "batch_size": 256,
                "learning_rate": 0.001,
                "dataset": "CIFAR-10",
                "architecture": "ResNeTiny",
                "loss": "BCS (SIGReg)"
            }
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_train_transforms()

    base_train_dataset = CIFAR10(root="./", train=True, download=True, transform=None)
    train_dataset = ImageDataset(base_train_dataset, transform, num_crops=2)
    views_visualization(base_train_dataset, num_images=20, filename="views_visualization.png")

    val_dataset = CIFAR10(root="./", train=False, download=True, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    model = ResNet().to(device)
    linear_probe = LinearProbe(feature_dim=128, num_classes=10).to(device)

    model_params = sum(p.numel() for p in model.parameters())
    print(f"Model Params: {model_params}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    probe_optimizer = optim.Adam(linear_probe.parameters(), lr=0.001)
    loss_fn = BCS(num_slices=128, lmbd=5.0)

    for epoch in range(0,300):
        train_ssl_loss, train_linear_loss, last_batch_features = train_epoch(model, linear_probe, train_loader, optimizer, probe_optimizer, device, epoch, loss_fn)
        val_acc, val_loss = evaluate_linear_probe(model, linear_probe, val_loader, device)

        print(f"Val Loss: {val_loss} -- Val Acc: {val_acc}")

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/ssl_loss": train_ssl_loss,
                "train/linear_loss": train_linear_loss,
                "val/loss": val_loss,
                "val/accuracy": val_acc
            })

    if use_wandb:
        wandb.finish()

    checkpoint = {
        'backbone_state_dict': model.state_dict(),
        'linear_probe_state_dict': linear_probe.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    checkpoint_path = "cifar10_ssl_checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=str, default="False")
    args = parser.parse_args()
    
    main(args)

