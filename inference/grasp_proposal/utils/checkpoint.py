# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os

import torch
import collections


class CheckPointer(object):
    def __init__(
            self,
            model,
            optimizer=None,
            scheduler=None,
            save_dir="",
            logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            self.logger.warning("No save directory specified. Can not save check point")
            return

        data = {"model": self.model.state_dict()}
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)
        file = os.path.join(self.save_dir, "last_checkpoint")
        with open(file, "w") as f:
            f.write(save_file)

    def load(self, filename=None, resume=True):
        if resume and self.has_checkpoint():
            # override argument with existing checkpoint
            filename = self.get_check_point_path()
        if not filename:
            # no checkpoint could be found
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("Loading checkpoint from {}".format(filename))
        checkpoint = torch.load(filename, map_location=torch.device("cpu"))
        self.model.load_state_dict(self._compatible_from_old_version(checkpoint.pop("model")), True)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(filename))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(filename))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        return checkpoint

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return os.path.exists(save_file)

    def get_check_point_path(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            last_saved = ""
            self.logger.warning("Last check point indicator file not exist, please check {}".format(
                os.path.join(self.save_dir, "last_checkpoint")))
        return last_saved

    @staticmethod
    def _compatible_from_old_version(old_model: collections.OrderedDict):
        new_model = collections.OrderedDict()
        for key, value in old_model.items():
            if key.startswith("module."):
                new_key = key[7:]
            else:
                new_key = key
            new_model.update({new_key: value})
        return new_model
