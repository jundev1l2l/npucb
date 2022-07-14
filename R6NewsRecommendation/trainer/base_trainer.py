import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from attrdict import AttrDict
from torch.optim import Adam
from tqdm import tqdm

from util.base_config import BaseConfig


class BaseTrainerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "lr": float,
            "num_epochs": int,
            "loss": str,
            "clip_loss": float,
            "val_freq": int,
            "save_freq": int
        }


class BaseTrainer: # todo: wandb 추가
    def __init__(self, config, model, dataloader_dict, logger, save_dir, rank, debug):
        self.lr = config.lr
        self.batch_size = dataloader_dict["train"].batch_size
        self.loss = config.loss
        self.clip_loss = config.clip_loss
        self.num_epochs = config.num_epochs
        self.val_freq = config.val_freq
        self.save_freq = config.save_freq    

        self.model = model
        self.optimizer = Adam(params=self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        self.train_dataloader = dataloader_dict["train"]
        self.val_dataloader = dataloader_dict["val"]
    
        self.logger = logger
        self.save_dir = save_dir
        self.rank = rank
        
        self.debug = debug
        if self.debug:
            self.num_epochs = 1
            self.val_freq = 1
            self.save_freq = 1

    def train(self):
        self.model.train().cuda()
        self.best_model = self.model
        self.best_val_loss = 1e8

        """
        Training Setting
        """
        self.logger.info("Training Setting")
        self.logger.info("")
        self.logger.info(f"Model {self.model.__class__}")
        self.logger.info(f"Batch Size {self.batch_size}")
        self.logger.info(f"Num Epochs {self.num_epochs}")
        self.logger.info(f"Optimizer {self.optimizer.__class__}")
        self.logger.info(f"Learning Rate {self.lr}")
        self.logger.info("")

        self.logger.info("Start Training")
        self.logger.info("")
        self.train_loss_list = []
        self.val_loss_list = []
        best_val_loss = 1e8

        for epoch in range(self.num_epochs):

            self.logger.info(f"Epoch {epoch + 1}")

            """
            Epoch Loop (Train)
            """
            epoch_train_loss = self.train_epoch()
            self.train_loss_list.append(epoch_train_loss)
            self.logger.info(f"[{epoch + 1:03d}/{self.num_epochs:03d}] Train Loss: {epoch_train_loss:.4f}")
            if self.rank <= 0:
                wandb.log({f"Loss_Train": epoch_train_loss}, step=epoch)

            """
            Epoch Loop (Val)
            """
            if (epoch + 1) % self.val_freq == 0:
                with torch.no_grad():
                    epoch_val_loss = self.val_epoch()
                self.val_loss_list.append(epoch_val_loss)
                self.logger.info(f"[{epoch + 1:03d}/{self.num_epochs:03d}] Val Loss: {epoch_val_loss:.4f}")
                self.logger.info("")
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    self.best_model = self.model
                if self.rank <= 0:
                    wandb.log({f"Loss_Validation": epoch_val_loss}, step=epoch)

            """
            Epoch Loop (Save)
            """
            if (epoch + 1) % self.save_freq == 0:
                self.save_model()
                self.save_loss()
                self.plot_loss()
                self.logger.info("")

        """
        Training Summary
        """
        self.save_model()
        self.save_loss()
        self.plot_loss()
        self.logger.info("")

    def train_epoch(self):
        self.model.train()
        self.train_bar = tqdm(self.train_dataloader)
        loss = .0
        for idx, batch in enumerate(self.train_bar):
            """
            minibatch loop (train)
            """
            self.train_bar.set_description(f"Batch Loss: {loss / (idx + 1e-3):.2f}")
            self.optimizer.zero_grad()
            batch_loss = self.get_batch_loss(batch)
            batch_loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss += batch_loss.item()

        return loss / len(self.train_dataloader)

    def val_epoch(self):
        self.model.eval()
        self.val_bar = tqdm(self.val_dataloader)
        loss = .0
        for idx, data in enumerate(self.val_bar):
            """
            minibatch loop (val)
            """
            self.val_bar.set_description(f"Batch Loss: {loss / (idx + 1e-3):.2f}")
            batch_loss = self.get_batch_loss(data)
            loss += batch_loss.item()

        return loss / len(self.val_dataloader)

    def eval(self, use_best=True):
        if use_best:
            self.model = self.best_model
        val_loss = self.val_epoch()
        if self.rank <= 0:
            wandb.run.summary[f"Loss_Validation"] = val_loss

    def get_batch_loss(self, batch):
        batch_dict = self.get_batch_dict(batch)
        outs = self.model(batch=batch_dict, loss=self.loss, clip_loss=self.clip_loss)
        batch_loss = outs.loss

        return batch_loss

    def get_batch_dict(self, batch):
        x, y = list(map(lambda x: x.cuda(), batch))
        batch_dict = AttrDict({"x": x, "y": y})

        return batch_dict

    def save_model(self):
        model_save = os.path.join(self.save_dir, "ckpt.tar")
        torch.save(self.best_model.module.state_dict() if self.rank >= 0 else self.best_model.state_dict(), model_save)
        self.logger.info(f"Best Model saved at {model_save}")

    def save_loss(self):
        train_loss_save = os.path.join(self.save_dir, "train_loss.npy")
        np.save(train_loss_save, np.array(self.train_loss_list))
        self.logger.info(f"Training Loss List saved at {train_loss_save}")

        val_loss_save = os.path.join(self.save_dir, "val_loss.npy")
        np.save(val_loss_save, np.array(self.val_loss_list))
        self.logger.info(f"Validation Loss List saved at {val_loss_save}")

    def plot_loss(self):
        train_epochs = list(range(self.val_freq, len(self.train_loss_list)))
        val_epochs = list(range(self.val_freq, (len(self.val_loss_list) + 1) * self.val_freq, self.val_freq))
        plt.clf()

        plt.title("Train Loss Curve")
        plt.plot(train_epochs, self.train_loss_list[self.val_freq:], color="black", label="train")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        train_loss_curve_save = os.path.join(self.save_dir, "train_curve.png")
        plt.savefig(train_loss_curve_save)
        self.logger.info(f"Train Loss Curve saved at {train_loss_curve_save}")
        plt.clf()

        plt.title("Validation Loss Curve")
        plt.plot(val_epochs, self.val_loss_list, color="red", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        val_loss_curve_save = os.path.join(self.save_dir, "val_curve.png")
        plt.savefig(val_loss_curve_save)
        self.logger.info(f"Validation Loss Curve saved at {val_loss_curve_save}")
        plt.clf()

        plt.title("Train & Validation Loss Curves")
        plt.plot(train_epochs, self.train_loss_list[self.val_freq:], color="black", label="train")
        plt.plot(val_epochs, self.val_loss_list, color="red", label="val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        both_loss_curve_save = os.path.join(self.save_dir, "both_curves.png")
        plt.savefig(both_loss_curve_save)
        self.logger.info(f"Train & Validation Loss Curves saved at {both_loss_curve_save}")
        plt.clf()
