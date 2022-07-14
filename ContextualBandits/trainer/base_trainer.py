import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from attrdict import AttrDict
from torch.optim import Adam
from tqdm import tqdm

from util.base_config import BaseConfig
from util.misc import get_circum_points


class BaseTrainerConfig(BaseConfig):
    def set_subconfig_class_hash(self):
        self.subconfig_class_hash = {
            "name": str,
            "lr": float,
            "num_epochs": int,
            "loss": str,
            "clip_loss": float,
            "val_freq": int,
            "save_freq": int,
            "plot_freq": int,
            "plot_grid_size": int,
        }


class BaseTrainer:
    def __init__(self, config, model, feature_extractor, dataloader_dict, logger, save_dir, rank, debug):
        self.lr = config.lr
        self.batch_size = dataloader_dict["train"].batch_size
        self.loss = config.loss
        self.clip_loss = config.clip_loss
        self.num_epochs = config.num_epochs
        self.val_freq = config.val_freq
        self.save_freq = config.save_freq
        self.plot_freq = config.plot_freq if hasattr(config, "plot_freq") else -1
        self.plot_grid_size = config.plot_grid_size if hasattr(config, "plot_grid_size") else 50

        self.model = model or feature_extractor
        self.feature_extractor = feature_extractor
        
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
        self.model.train()
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

        for self.epoch in range(self.num_epochs):

            self.logger.info(f"Epoch {self.epoch + 1}")

            """
            Epoch Loop (Train)
            """
            epoch_train_loss = self.train_epoch()
            self.train_loss_list.append(epoch_train_loss)
            self.logger.info(f"[{self.epoch + 1:03d}/{self.num_epochs:03d}] Train Loss: {epoch_train_loss:.4f}")
            if self.rank <= 0:
                wandb.log({f"Loss_Train": epoch_train_loss}, step=self.epoch)

            """
            Epoch Loop (Val)
            """
            if (self.epoch + 1) % self.val_freq == 0:
                with torch.no_grad():
                    epoch_val_loss = self.val_epoch()
                self.val_loss_list.append(epoch_val_loss)
                self.logger.info(f"[{self.epoch + 1:03d}/{self.num_epochs:03d}] Val Loss: {epoch_val_loss:.4f}")
                self.logger.info("")
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    self.best_model = self.model
                if self.rank <= 0:
                    wandb.log({f"Loss_Validation": epoch_val_loss}, step=self.epoch)
            
            """
            Epoch Loop (Plot)
            """
            if (self.plot_freq > 0) and ((self.epoch + 1) % self.plot_freq == 0):
                with torch.no_grad():
                    self.plot_epoch()

            """
            Epoch Loop (Save)
            """
            if (self.epoch + 1) % self.save_freq == 0:
                self.save_model()
                # self.save_loss()
                # self.plot_loss()
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
            data_sampler = self.val_dataloader.dataset.data_sampler
            data_sampler_name = type(data_sampler[0]).__name__ if isinstance(data_sampler, list) else type(data_sampler[0]).__name__

        return loss / len(self.val_dataloader)
    
    def plot_epoch(self):
        self.best_model.eval()
        self.plot_bar = tqdm(self.val_dataloader)
        for idx, data in enumerate(self.plot_bar):
            if idx not in [0,1,2,3,4]:
                break
            """
            minibatch loop (plot)
            """
            batch_loss = self.get_batch_loss(data)
            data_sampler = self.val_dataloader.dataset.data_sampler
            data_sampler_name = type(data_sampler[0]).__name__ if isinstance(data_sampler, list) else type(data_sampler[0]).__name__
            if (data_sampler_name == "WheelDataSampler"):
                self.plot_wheel(data, batch_loss.item(), idx)

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
        x, y = list(map(lambda x: x.cuda() if self.rank >= 0 else x.cpu(), batch))
        
        if self.feature_extractor is not None:
            x = self.feature_extractor.encode(x)
        
        batch_dict = AttrDict({"x": x, "y": y,})

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
        plt.xlabel("self.epoch")
        plt.ylabel("Loss")
        plt.legend()
        train_loss_curve_save = os.path.join(self.save_dir, "train_curve.png")
        plt.savefig(train_loss_curve_save)
        self.logger.info(f"Train Loss Curve saved at {train_loss_curve_save}")
        plt.clf()

        plt.title("Validation Loss Curve")
        plt.plot(val_epochs, self.val_loss_list, color="red", label="val")
        plt.xlabel("self.epoch")
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

    def plot_wheel(self, data, loss, idx):
        batch_dict = self.get_batch_dict(data)
        grid_size = self.plot_grid_size

        x, y = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size))
        samples = np.array(list(zip(x.reshape(-1), y.reshape(-1))))
        samples_repeated = np.repeat(np.expand_dims(samples, axis=0), 5, axis=0)
        arms = np.arange(5).reshape(-1, 1)
        arms_repeated = np.repeat(np.expand_dims(arms, axis=1), samples_repeated.shape[1], axis=1)
        xt = torch.FloatTensor(np.concatenate([samples_repeated, arms_repeated], axis=-1).reshape(1,-1,3))
        xc = batch_dict.xc[0].unsqueeze(0)
        yc = batch_dict.yc[0].unsqueeze(0)
        delta = batch_dict.delta[0,0,0]
        
        preds = self.best_model.predict(xc, yc, xt)
        mean = (preds.loc).reshape(5, grid_size**2, 1)
        std = (preds.scale).reshape(5, grid_size**2, 1)
        ucbs = mean + 1.0 * std

        outer_radius = 0.5 * grid_size
        inner_radius = 0.5 * grid_size * delta
        outer_circle = get_circum_points(outer_radius, grid_size ** 2) + [outer_radius, outer_radius]
        inner_circle = get_circum_points(inner_radius, grid_size ** 2) + [outer_radius, outer_radius]
        left_x_axis = np.stack([np.linspace(0, outer_radius - inner_radius, grid_size ** 2), outer_radius * np.ones(grid_size ** 2)], axis=-1)
        right_x_axis = np.stack([np.linspace(outer_radius + inner_radius, grid_size, grid_size ** 2), outer_radius * np.ones(grid_size ** 2)], axis=-1)
        lower_y_axis = np.stack([outer_radius * np.ones(grid_size ** 2), np.linspace(0, outer_radius - inner_radius, grid_size ** 2)], axis=-1)
        upper_y_axis = np.stack([outer_radius * np.ones(grid_size ** 2), np.linspace(outer_radius + inner_radius, grid_size, grid_size ** 2)], axis=-1)

        boundary = [
            outer_circle, 
            inner_circle, 
            left_x_axis,
            right_x_axis,
            lower_y_axis,
            upper_y_axis,
        ]

        for a, (m, s, u) in enumerate(zip(mean, std, ucbs)):
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            plt.suptitle(f"Reward Prediction of Arm {a} at Epoch {self.epoch} (Loss={round(loss, 3)})", size=40)
            data_dict = {
                "Mean": m,
                "Std": s,
                "Ucb": u,
            }
            for idx, (name, data) in enumerate(data_dict.items()):
                axes[idx].set_title(name, size=30)
                c = axes[idx].pcolor(data.reshape(grid_size, grid_size), cmap="jet", )
                for b in boundary:
                    axes[idx].plot(b[:, 0], b[:, 1], "k-")
                plt.sca(axes[idx])
                num_ticks = 10
                plt.xticks(np.arange(num_ticks + 1) * int(grid_size / num_ticks), np.linspace(-1, 1, num_ticks + 1).round(1))
                plt.yticks(np.arange(num_ticks + 1) * int(grid_size / num_ticks), np.linspace(-1, 1, num_ticks + 1).round(1))
                plt.colorbar(c, ax=axes[idx])

            plot_dir = os.path.join(self.save_dir, "plot")
            os.makedirs(plot_dir, exist_ok=True)
            image_file = os.path.join(plot_dir, f"RewardPrediction_Arm={a}_Idx={idx}_Epoch={self.epoch}.png")
            plt.tight_layout()
            plt.savefig(image_file)
            plt.close()
            wandb.log(data={
                f"{self.model.name}/RewardPrediction/Idx={idx}/Arm={a}": wandb.Image(image_file),
            }, step=self.epoch)
