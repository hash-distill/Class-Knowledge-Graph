import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SAGENet(nn.Module):
    """
    SAGE-Net: Student Attention and Engagement Evaluation Network
    学生注意力与参与度评估网络

    功能:
    - 头部姿态估计 (pitch, yaw, roll)
    - 视线方向回归 (x, y)
    - 表情分类 (7类)
    - 参与度评分融合 (0-1)
    """

    def __init__(self, expression_classes: int = 7):
        """
        初始化SAGE-Net

        Args:
            expression_classes: 表情类别数，默认7类
        """
        super().__init__()

        # 共享Backbone: MobileNetV3-Large (轻量高效)
        backbone = models.mobilenet_v3_large(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.feature_dim = 960  # MobileNetV3-large输出维度

        # 任务1: 头部姿态回归 - 预测3D旋转角度
        self.head_pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 3)  # [pitch, yaw, roll]
        )

        # 任务2: 视线方向回归 - 预测2D归一化坐标
        self.gaze_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),  # [x, y]
            nn.Tanh()  # 归一化到[-1, 1]
        )

        # 任务3: 表情分类 - 7类情绪
        self.expression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, expression_classes)
        )

        # 任务4: 参与度评分 - 多特征融合
        self.engagement_fusion = nn.Sequential(
            nn.Linear(self.feature_dim + 3 + 2, 256),  # 特征+姿态+视线
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出[0, 1]概率
        )

        # 初始化权重
        self._init_weights()

        print(f"SAGE-Net初始化完成 | 特征维度: {self.feature_dim}")

    def _init_weights(self):
        """初始化头部网络权重"""
        for head in [self.head_pose_head, self.gaze_head, self.expression_head, self.engagement_fusion]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入图像 [B, 3, 224, 224]

        Returns:
            字典包含所有任务输出
        """
        batch_size = x.size(0)

        # 共享特征提取
        features = self.backbone(x)  # [B, 960, 7, 7]

        # 并行多任务预测
        head_pose = self.head_pose_head(features)  # [B, 3]
        gaze = self.gaze_head(features)  # [B, 2]
        expression = self.expression_head(features)  # [B, 7]

        # 特征池化用于融合
        pooled_feat = torch.adaptive_avg_pool2d(features, 1).view(batch_size, -1)  # [B, 960]

        # 参与度多特征融合
        engagement_feat = torch.cat([pooled_feat, head_pose, gaze], dim=1)  # [B, 960+3+2]
        engagement = self.engagement_fusion(engagement_feat)  # [B, 1]

        return {
            "head_pose": head_pose,
            "gaze": gaze,
            "expression": expression,
            "engagement": engagement,
            "features": pooled_feat
        }


class ActionRecognizer(nn.Module):
    """
    时序行为识别网络
    识别学生动作：举手、记笔记、趴下、站立等
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        """
        初始化行为识别器

        Args:
            num_classes: 动作类别数
            pretrained: 是否使用预训练权重
        """
        super().__init__()

        try:
            from pytorchvideo.models import slowfast_r50
            self.slowfast = slowfast_r50(pretrained=pretrained)

            # 修改输出层
            self.slowfast.blocks[-1].proj = nn.Linear(2304, num_classes)

            print(f"ActionRecognizer初始化完成 | 类别数: {num_classes}")
        except ImportError:
            print("pytorchvideo未安装，使用备用3D-ResNet")
            self.slowfast = self._build_3d_resnet(num_classes)

    def _build_3d_resnet(self, num_classes: int):
        """构建简化的3D ResNet"""
        from torchvision.models.video import r3d_18
        model = r3d_18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 视频片段 [B, 3, 16, 224, 224]

        Returns:
            动作 logits [B, num_classes]
        """
        return self.slowfast(x)


# 模型测试代码
if __name__ == "__main__":
    # 测试SAGE-Net
    sage_net = SAGENet(expression_classes=7)
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        outputs = sage_net(dummy_input)

    print("\nSAGE-Net输出形状:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # 测试行为识别
    action_rec = ActionRecognizer(num_classes=4)
    dummy_video = torch.randn(1, 3, 16, 224, 224)

    with torch.no_grad():
        action_out = action_rec(dummy_video)

    print(f"\nActionRecognizer输出: {action_out.shape}")