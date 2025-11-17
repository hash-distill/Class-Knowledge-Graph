import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
from pathlib import Path
import argparse

from student_engagement.models.multitask_net import SAGENet


class SAGEDataset(Dataset):
    """SAGE-Net训练数据集"""

    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.samples = []

        # 加载标注文件
        label_file = self.data_dir / "labels.json"
        if label_file.exists():
            with open(label_file, "r") as f:
                self.samples = json.load(f)

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像
        img_path = self.data_dir / "images" / sample["image_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        # 归一化
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # 加载标签
        head_pose = torch.tensor(sample["head_pose"], dtype=torch.float32)  # [3]
        gaze = torch.tensor(sample["gaze"], dtype=torch.float32)  # [2]
        expression = torch.tensor(sample["expression"], dtype=torch.long)  # [1]
        engagement = torch.tensor(sample["engagement"], dtype=torch.float32)  # [1]

        return image, {
            "head_pose": head_pose,
            "gaze": gaze,
            "expression": expression,
            "engagement": engagement
        }


class SAGELoss(nn.Module):
    """SAGE-Net多任务损失函数"""

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # 头部姿态损失
        pose_loss = self.mse_loss(predictions["head_pose"], targets["head_pose"])

        # 视线损失
        gaze_loss = self.mse_loss(predictions["gaze"], targets["gaze"])

        # 表情分类损失
        expression_loss = self.cross_entropy_loss(predictions["expression"], targets["expression"])

        # 参与度损失
        engagement_loss = self.mse_loss(predictions["engagement"].squeeze(), targets["engagement"])

        # 总损失（加权）
        total_loss = pose_loss * 0.3 + gaze_loss * 0.2 + expression_loss * 0.3 + engagement_loss * 0.2

        return total_loss, {
            "pose_loss": pose_loss.item(),
            "gaze_loss": gaze_loss.item(),
            "expression_loss": expression_loss.item(),
            "engagement_loss": engagement_loss.item(),
            "total_loss": total_loss.item()
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        # 前向传播
        predictions = model(images)

        # 计算损失
        loss, loss_dict = criterion(predictions, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss_dict["total_loss"]
        num_batches += 1

        

    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            predictions = model(images)
            _, loss_dict = criterion(predictions, targets)

            total_loss += loss_dict["total_loss"]
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="SAGE-Net训练脚本")
    parser.add_argument("--data_dir", required=True, help="训练数据目录")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--device", default="cuda", help="训练设备")
    parser.add_argument("--save_dir", default="outputs", help="模型保存目录")

    args = parser.parse_args()

    # 创建设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    train_dataset = SAGEDataset(Path(args.data_dir) / "train")
    val_dataset = SAGEDataset(Path(args.data_dir) / "val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 创建模型
    model = SAGENet(expression_classes=7).to(device)
    criterion = SAGELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, save_dir / "sage_net_best.pth")
            

        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_dir / f"sage_net_epoch_{epoch + 1}.pth")

    


if __name__ == "__main__":
    main()