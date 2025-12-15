from torch.utils.data import DataLoader
from geoseg.datasets.inbdc_dataset import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from geoseg.losses import *
from geoseg.models.NetModel_ConvNextDWT import Model_ConvNextDWT
import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR

max_epoch = 90
ignore_index = 255
train_batch_size = 12
val_batch_size = 12

lr = 0.0005
weight_decay = 0.0005
backbone_lr = 0.0001
backbone_weight_decay = 0.0005

accumulate_n = 2
num_classes = len(CLASSES)
classes = CLASSES

weights_name = "model"
weights_path = "/remote-home/public2/NetModel/results/inbdc/ConvNextDWT/4".format(weights_name)
test_weights_name = weights_name
log_name = "/remote-home/public2/NetModel/results/inbdc/ConvNextDWT/4".format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None


net = Model_ConvNextDWT(n_class=num_classes)
loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
use_aux_loss = False

# define the dataloader

train_dataset = inbdcBuildingDataset(data_root= "/remote-home/public2/NetModel/dataset/inbdc/train/", mode='train', mosaic_ratio=0.25, transform=train_aug)
val_dataset = inbdcBuildingDataset(data_root= "/remote-home/public2/NetModel/dataset/inbdc/test/", mode='val', transform=val_aug)
test_dataset = inbdcBuildingDataset(data_root="/remote-home/public2/NetModel/dataset/inbdc/test/", mode='val', transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)

main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

warmup_epochs = 5
warmup_scheduler = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs)

lr_scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, main_scheduler],
    milestones=[warmup_epochs]
)