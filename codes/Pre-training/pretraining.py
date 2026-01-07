import csv
import h5py
import torch
import torch.nn as nn
import random
import numpy as np
import os
import shutil
import pandas as pd
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, Subset, random_split
import torch.optim as optim
import time
from tqdm import tqdm
from torch.optim import lr_scheduler
from transformers import ViTFeatureExtractor, AutoImageProcessor, ViTMAEConfig, ViTMAEModel, ViTMAEForPreTraining
from torchvision.datasets import ImageFolder
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.utilities import rank_zero_only

DEVICE_NUM = torch.cuda.device_count()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(DEVICE_NUM)])

SEED = 42
DATA_DIR = "../../0.data/pretrain_nucleus_image_all_16M.hdf5"
BATCH_SIZE = 400 *2
NUM_EPOCHS = 70
LEARNINGRATE = 0.0001
PROJECT_NAME = 'Nuspire_Pretraining_V5'

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.RandomResizedCrop((112, 112), scale=(0.5625, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.21869252622127533], std=[0.1809280514717102])
])

configuration = ViTMAEConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=112,
        patch_size=8,
        num_channels=1,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=1024,
        mask_ratio=0.75,
        norm_pix_loss=False
)

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        self.hdf5_file = h5py.File(hdf5_path, 'r', rdcc_nbytes=10*1024**3, rdcc_w0=0.0, rdcc_nslots=10007)
        self.images = self.hdf5_file['images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        
        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)
        
        return img

    def __del__(self):
        self.hdf5_file.close()

class NucleusDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

    def setup(self, stage=None):
        
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=16, pin_memory=True, prefetch_factor=5)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size * 3, num_workers=16, pin_memory=True, prefetch_factor=5)

class ViTMAEPreTraining(pl.LightningModule):
    def __init__(self, configuration):
        super().__init__()
        self.model = ViTMAEForPreTraining(configuration) 
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.to(self.device)
        outputs = self.model(x)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

   
    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.to(self.device)
        outputs = self.model(x)
        loss = outputs.loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss       
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNINGRATE)
        warmup_epochs = 10
        warmup_factor = lambda epoch: epoch / warmup_epochs if epoch < warmup_epochs else 1
        scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_factor)
        scheduler_regular = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_regular], milestones=[warmup_epochs]),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]

class EpochLoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get('train_loss')
        val_loss = trainer.callback_metrics.get('val_loss')
        if train_loss is not None and val_loss is not None:
            trainer.logger.experiment.add_scalars(
                "Epoch/Loss",
                {'Train Loss': train_loss, 'Validation Loss': val_loss},
                trainer.current_epoch
            )

class SaveEpochModelCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    
    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        path = trainer.checkpoint_callback.dirpath
        epoch = trainer.current_epoch
        pl_module.model.save_pretrained(f'{path}/epoch{epoch}')

dataset = HDF5Dataset(hdf5_path=DATA_DIR, transform=transform)

data_module = NucleusDataModule(dataset, BATCH_SIZE)

epoch_logging_callback = EpochLoggingCallback()

save_epoch_model_callback = SaveEpochModelCallback()

progress_bar = RichProgressBar()

logger = TensorBoardLogger(save_dir=f'./{PROJECT_NAME}_outputs', name="tensorboard")

best_model_callback = ModelCheckpoint(
    dirpath=f'./{PROJECT_NAME}_outputs/model',
    filename='{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
    monitor='val_loss'
)

lr_monitor = LearningRateMonitor(logging_interval='epoch')

trainer = Trainer(
    max_epochs=NUM_EPOCHS,
    devices=DEVICE_NUM,  # 设置使用的设备数量
    accelerator='gpu',  # 指定使用GPU
    strategy='ddp',
    logger=logger,
    callbacks=[lr_monitor,
               progress_bar, 
               epoch_logging_callback, 
               save_epoch_model_callback, 
               best_model_callback]
)

# 设置随机种子
pl.seed_everything(SEED, workers=True)

model = ViTMAEPreTraining(configuration,)
trainer.fit(model, data_module)


