import os
import logging

mainlogger = logging.getLogger('mainlogger')

import torch
import pandas as pd

from omegaconf import OmegaConf
from collections import OrderedDict


def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'),
                        exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}),
                       os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo


def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None


def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "batch_logger": {
            "target": "unifolm_wma.utils.callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            }
        },
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "unifolm_wma.utils.callbacks.CUDACallback",
        },
    }

    # Optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"][
            "monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info(
            'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.'
        )
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {
                "target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                'params': {
                    "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                    "filename": "{epoch}-{step}",
                    "verbose": True,
                    'save_top_k': -1,
                    'every_n_train_steps': 10000,
                    'save_weights_only': True
                }
            }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg


def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                "name": "testtube",
                "save_dir": logdir,
            }
        },
    }
    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg


def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg


# def load_checkpoints(model, model_cfg):
#     if check_config_attribute(model_cfg, "pretrained_checkpoint"):
#         pretrained_ckpt = model_cfg.pretrained_checkpoint
#         assert os.path.exists(
#             pretrained_ckpt
#         ), "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
#         mainlogger.info(">>> Load weights from pretrained checkpoint")

#         pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
#         try:
#             if 'state_dict' in pl_sd.keys():
#                 model.load_state_dict(pl_sd["state_dict"], strict=False)
#                 mainlogger.info(
#                     ">>> Loaded weights from pretrained checkpoint: %s" %
#                     pretrained_ckpt)
#             else:
#                 # deepspeed
#                 new_pl_sd = OrderedDict()
#                 for key in pl_sd['module'].keys():
#                     new_pl_sd[key[16:]] = pl_sd['module'][key]
#                 model.load_state_dict(new_pl_sd, strict=False)
#         except:
#             model.load_state_dict(pl_sd)
#     else:
#         mainlogger.info(">>> Start training from scratch")

#     return model

def _remap_cond_weight(v, t, dsed, n_obs, rgb_dim, old_D, new_D, keep):
    """v: [out, Cin_old] -> t: [out, Cin_new]. Layout: [t_emb | (rgb|agent_pos)*n_obs | coda]."""
    Fo, Fn = rgb_dim + old_D, rgb_dim + new_D
    new = torch.zeros_like(t)
    new[:, :dsed] = v[:, :dsed]
    for i in range(n_obs):
        so, sn = dsed + i * Fo, dsed + i * Fn
        new[:, sn:sn + rgb_dim] = v[:, so:so + rgb_dim]
        k = min(keep, old_D)
        new[:, sn + rgb_dim:sn + rgb_dim + k] = v[:, so + rgb_dim:so + rgb_dim + k]
    to, tn = dsed + n_obs * Fo, dsed + n_obs * Fn
    assert v.shape[1] - to == t.shape[1] - tn, "coda del cond diversa"
    new[:, tn:] = v[:, to:]
    return new


def load_state_dict_resized(model, sd, keep=14, old_D=16, new_D=26,
                            dsed=128, n_obs=2):
    msd = model.state_dict()
    rgb_dim = model.model.diffusion_model.action_unet.obs_encoder.output_shape()[-1] - new_D
    out, resized, skipped = {}, [], []
    for k, v in sd.items():
        t = msd.get(k)
        if t is None or v.shape == t.shape:
            out[k] = v
        elif 'cond_encoder' in k and v.ndim == 2 and v.shape[0] == t.shape[0] \
                and t.shape[1] - v.shape[1] == n_obs * (new_D - old_D):
            out[k] = _remap_cond_weight(v, t, dsed, n_obs, rgb_dim, old_D, new_D, keep)
            resized.append((k, 'cond', tuple(v.shape), tuple(t.shape)))
        elif v.ndim == t.ndim and all(s == ts or (s == old_D and ts == new_D)
                                      for s, ts in zip(v.shape, t.shape)):
            new = torch.zeros_like(t)
            idx = tuple(slice(None) if s == ts else slice(0, keep)
                        for s, ts in zip(v.shape, t.shape))
            new[idx] = v[idx].to(new.dtype)
            out[k] = new
            resized.append((k, 'chan', tuple(v.shape), tuple(t.shape)))
        else:
            skipped.append((k, tuple(v.shape), tuple(t.shape)))
    for r in resized: mainlogger.info(f">>> resized: {r}")
    for s in skipped: mainlogger.info(f">>> SKIPPED (random init): {s}")
    return model.load_state_dict(out, strict=False)


def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), \
            "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        if 'state_dict' in pl_sd:
            sd = pl_sd['state_dict']
        else:  # deepspeed
            sd = OrderedDict((k[16:], v) for k, v in pl_sd['module'].items())
        load_state_dict_resized(model, sd, keep=14)
        mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s" % pretrained_ckpt)
    else:
        mainlogger.info(">>> Start training from scratch")
    return model


def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_num_parameters(model):
    models = [('World Model', model.model.diffusion_model),
              ('Action Head', model.model.diffusion_model.action_unet),
              ('State Head', model.model.diffusion_model.state_unet),
              ('Total Trainable', model),
              ('Total', model)]

    data = []
    for index, (name, model) in enumerate(models):
        if name == "Total Trainable":
            total_params = count_trainable_parameters(model)
        else:
            total_params = count_parameters(model)
        if total_params < 0.1e9:
            total_params_value = round(total_params / 1e6, 2)
            unit = 'M'
        else:
            total_params_value = round(total_params / 1e9, 2)
            unit = 'B'

        data.append({
            'Model Name': name,
            'Params': f"{total_params_value} {unit}"
        })

    df = pd.DataFrame(data)
    print(df)
