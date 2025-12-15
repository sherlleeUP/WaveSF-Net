import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, inputs, targets):
        return self.ce(inputs, targets)


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


class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
   
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.sobel_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y, requires_grad=False)

    def forward(self, inputs, targets):
   
        targets = targets.unsqueeze(1).float()
        edges_t_x = self.sobel_x(targets)
        edges_t_y = self.sobel_y(targets)
        edges_t = torch.sqrt(edges_t_x ** 2 + edges_t_y ** 2)
        edges_t = (edges_t > 0).float()

      
        inputs = F.softmax(inputs, dim=1)
        pred = inputs.argmax(dim=1, keepdim=True).float()
        edges_p_x = self.sobel_x(pred)
        edges_p_y = self.sobel_y(pred)
        edges_p = torch.sqrt(edges_p_x ** 2 + edges_p_y ** 2)
        edges_p = (edges_p > 0).float()

       
        intersection = (edges_p * edges_t).sum(dim=(2, 3))
        union = edges_p.sum(dim=(2, 3)) + edges_t.sum(dim=(2, 3))
        dice = (2. * intersection) / (union + 1e-6)
        loss = 1 - dice.mean()
        return loss


class ShapeRegularizationLoss(nn.Module):
    def __init__(self):
        super(ShapeRegularizationLoss, self).__init__()
        
        self.laplacian = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        laplacian_kernel = torch.tensor([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.laplacian.weight = nn.Parameter(laplacian_kernel, requires_grad=False)

    def forward(self, inputs):
        # inputs: [B, C, H, W] after softmax
        pred = inputs.argmax(dim=1, keepdim=True).float()
        laplacian_pred = self.laplacian(pred)
        loss = torch.mean(torch.abs(laplacian_pred))
        return loss


class MultiScaleLoss(nn.Module):
    def __init__(self, num_classes, weight_ce=None, alpha=1.0, beta=1.0, gamma=0.5, delta=0.2):
        super(MultiScaleLoss, self).__init__()
        self.alpha = alpha  # Weight for Cross-Entropy Loss
        self.beta = beta  # Weight for Dice Loss
        self.gamma = gamma  # Weight for Boundary Loss
        self.delta = delta  # Weight for Shape Regularization Loss

        self.ce_loss = WeightedCrossEntropyLoss(weight=weight_ce)
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()
        self.shape_reg_loss = ShapeRegularizationLoss()

    def forward(self, predictions, targets):
        """
        predictions: tuple/list of [P1, P2, ..., PN], each P_i has shape [B, C, H_i, W_i]
        targets: [B, H, W]
        """
        total_loss = 0.0
        num_scales = len(predictions)
        for pred in predictions:
            
            pred_upsampled = F.interpolate(pred, size=(512, 512), mode='bilinear', align_corners=False)
           
            loss_ce = self.ce_loss(pred_upsampled, targets)
            loss_dice = self.dice_loss(pred_upsampled, targets)
            loss_boundary = self.boundary_loss(pred_upsampled, targets)
            loss_shape = self.shape_reg_loss(pred_upsampled)

          
            total_loss += self.alpha * loss_ce + self.beta * loss_dice + self.gamma * loss_boundary + self.delta * loss_shape

        
        total_loss /= num_scales
        return total_loss
