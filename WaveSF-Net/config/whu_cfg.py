from torch.utils.data import DataLoader
from geoseg.datasets.whubuilding_dataset import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from geoseg.losses import *
from geoseg.models.NetModel_ConvNextDWT import Model_ConvNextDWT
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR

max_epoch = 105
ignore_index = 255
train_batch_size = 12
val_batch_size = 12

lr = 0.000045
weight_decay = 0.00001
backbone_lr = 0.000045
backbone_weight_decay = 0.00001

accumulate_n = 2
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "model"
weights_path = "/remote-home/public2/NetModel/results/whu/ConvNextDWT/final".format(weights_name)
test_weights_name = weights_name
log_name = "/remote-home/public2/NetModel/results/whu/ConvNextDWT/final".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None


class StableCombinedLoss(nn.Module):
    def __init__(self, num_classes, ignore_index):
        super(StableCombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.epsilon = 1e-7
        self.current_epoch = 0
        self.use_dice = False

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        self.use_dice = (epoch >= 10)

    def stable_dice_loss(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = torch.clamp(y_pred, self.epsilon, 1 - self.epsilon)

        y_true_onehot = torch.nn.functional.one_hot(
            y_true.long(), num_classes=self.num_classes
        ).permute(0, 3, 1, 2).float()

        intersection = torch.sum(y_pred * y_true_onehot, dim=(2, 3))
        union = torch.sum(y_pred, dim=(2, 3)) + torch.sum(y_true_onehot, dim=(2, 3))

        dice = (2.0 * intersection + self.epsilon) / (union + self.epsilon)

        return 1.0 - dice.mean()

    def forward(self, y_pred, y_true):
        ce = self.ce_loss(y_pred, y_true)

        if not self.use_dice:
            return ce

        dice = self.stable_dice_loss(y_pred, y_true)

        return ce + dice


loss = StableCombinedLoss(num_classes=num_classes, ignore_index=ignore_index)
use_aux_loss = False

net = Model_ConvNextDWT(n_class=num_classes)

train_dataset = WHUBuildingDataset(
    data_root="/remote-home/public2/NEW/data/whubuilding/train",
    mode='train',
    mosaic_ratio=0.15,
    transform=train_aug
)

val_dataset = WHUBuildingDataset(
    data_root="/remote-home/public2/NEW/data/whubuilding/val",
    mode='val',
    transform=val_aug
)

test_dataset = WHUBuildingDataset(
    data_root="/remote-home/public2/NEW/data/whubuilding/test",
    mode='val',
    transform=val_aug
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=train_batch_size,
    num_workers=4,
    pin_memory=True,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    drop_last=False
)

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)

base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

warmup_epochs = 5
lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[
        LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch - warmup_epochs)
    ],
    milestones=[warmup_epochs]
)


class GradientClippingCallback:
    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def on_before_optimizer_step(self, runner):
        torch.nn.utils.clip_grad_norm_(runner.model.parameters(), self.max_norm)


callbacks = {
    'gradient_clipping': GradientClippingCallback(max_norm=1.0)
}


class LossUpdateCallback:
    def on_epoch_start(self, runner):
        runner.criterion.update_epoch(runner.epoch)