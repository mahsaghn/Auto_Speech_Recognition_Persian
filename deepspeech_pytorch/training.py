import json

import hydra
from deepspeech_pytorch.checkpoint import GCSCheckpointHandler, FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, GCSCheckpointConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech, SequenceWise
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
import torch
from torch import nn
from omegaconf import MISSING


def train(cfg: DeepSpeechConfig):
    seed_everything(cfg.seed)
    print('1')
    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)
    print('2',cfg.trainer.checkpoint_callback)

    if cfg.trainer.checkpoint_callback:
        print(OmegaConf.get_type(cfg.checkpoint))
        if OmegaConf.get_type(cfg.checkpoint) is GCSCheckpointConfig:
            print('in if checkpoint')
            checkpoint_callback = GCSCheckpointHandler(
                cfg=cfg.checkpoint
            )
        else:
            cfg.checkpoint.mode = 'min'
            checkpoint_callback = FileCheckpointHandler(
                cfg=cfg.checkpoint
            )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint
    print('3')
    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        is_distributed=False
    )
    print('4')

    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect,
        num_classes=35
    )

    # if cfg.transfer_checkpoint != MISSING:
    print("Load Pretrained Model...")
    #chkp = torch.load(cfg.transfer_checkpoint)
    #model.load_state_dict(chkp['state_dict'])
    #fully_connected = nn.Sequential(
    #    nn.BatchNorm1d(cfg.model.hidden_size),
    #    nn.Linear(cfg.model.hidden_size, len(labels), bias=False)
    #)
    #model.fc = nn.Sequential(
    #    SequenceWise(fully_connected),
    #)

    print('5')

    trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        replace_sampler_ddp=False,
        callbacks=[checkpoint_callback] if cfg.trainer.checkpoint_callback else None,
    )
    print('6')

    trainer.fit(model, data_loader)
