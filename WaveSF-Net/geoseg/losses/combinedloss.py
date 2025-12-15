import torch
import torch.nn as nn
import torch.nn.functional as F
class EdgeLoss(nn.Module):
    """
    EdgeLoss 用于计算图像边缘的重建损失，通常用于分割任务中增强对边缘区域的关注。
    使用 Sobel 滤波器提取边缘，并通过均方误差 (MSE) 或交叉熵 (BCE) 计算边缘损失。
    """
    def __init__(self, ignore_index=None, reduction='mean', use_bce=False):
        """
        参数：
        - ignore_index (int, optional): 忽略某些标签值（例如背景或无效像素），默认 None。
        - reduction (str): 损失的归约方式，可以是 'mean', 'sum', 或 'none'，默认 'mean'。
        - use_bce (bool): 是否使用二值交叉熵损失（适用于二值边缘），默认 False（使用 MSE）。
        """
        super(EdgeLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.use_bce = use_bce

        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('sobel_x_weight', self.sobel_x)
        self.register_buffer('sobel_y_weight', self.sobel_y)

    def forward(self, predictions, targets):
        """
        计算 EdgeLoss。
        
        参数：
        - predictions (torch.Tensor): 模型预测结果，形状为 (batch_size, num_classes, height, width) 或 (batch_size, 1, height, width)。
        - targets (torch.Tensor): 真实标签，形状为 (batch_size, height, width)，值通常为 0 或 1（对于二值边缘）。

        返回：
        - loss (torch.Tensor): 计算得到的损失值。
        """
        if predictions.dim() == 4 and predictions.size(1) > 1:  
            predictions = F.softmax(predictions, dim=1).argmax(dim=1, keepdim=False).float()
        elif predictions.dim() == 4 and predictions.size(1) == 1:  
            predictions = torch.sigmoid(predictions).squeeze(1)  
        else:
            raise ValueError("predictions 形状不正确，应为 (batch_size, C, H, W) 或 (batch_size, 1, H, W)")

        targets = targets.float()

        batch_size, height, width = targets.size()

        pred_edges = self._compute_edges(predictions)
        true_edges = self._compute_edges(targets)

        if self.use_bce:
            loss = F.binary_cross_entropy_with_logits(pred_edges, true_edges, reduction='none')
        else:
            loss = F.mse_loss(pred_edges, true_edges, reduction='none')

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            loss = loss[mask]

        if self.reduction == 'mean' and loss.numel() > 0:
            loss = loss.mean()
        elif self.reduction == 'sum' and loss.numel() > 0:
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f"不支持的 reduction 模式: {self.reduction}")

        return loss

    def _compute_edges(self, image):
        """
        使用 Sobel 滤波器计算图像的边缘。
        
        参数：
        - image (torch.Tensor): 输入图像或掩码，形状为 (batch_size, height, width) 或 (batch_size, 1, height, width)。

        返回：
        - edges (torch.Tensor): 边缘强度，形状与输入相同。
        """
        if image.dim() == 3:
            image = image.unsqueeze(1)

        batch_size, _, height, width = image.size()

        image = image.float()

        padding = 1  
        sobel_x = self.sobel_x_weight.expand(1, 1, 3, 3).to(image.device)
        sobel_y = self.sobel_y_weight.expand(1, 1, 3, 3).to(image.device)

        edges_x = F.conv2d(image, sobel_x, padding=padding, groups=1)
        edges_y = F.conv2d(image, sobel_y, padding=padding, groups=1)

        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)

        return edges


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] after softmax
        # targets: [B, H, W]
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice.mean()
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(CombinedLoss, self).__init__()
        self.edge_loss = EdgeLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss()
        self.alpha = 0.5  # 权重，可以作为参数传入

    def forward(self, outputs, targets):
        edge_loss = self.edge_loss(outputs, targets)
        dice_loss = self.dice_loss(outputs, targets)
        return self.alpha * edge_loss + (1 - self.alpha) * dice_loss