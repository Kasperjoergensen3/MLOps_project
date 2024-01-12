from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from sklearn.manifold import TSNE


class ViTWithoutClassification(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.ViT = vit_model.ViT
        # remove last layer
        self.ViT = nn.Sequential(*list(self.ViT.children())[:-1])
        self.head = vit_model.head
        # identity layer

    def forward(self, x):
        x = self.head(x)
        x = self.ViT(x)
        return torch.mean(x.last_hidden_state, dim=1)


class PlotLogger(Callback):
    def __init__(self, model, data_module, config=None):
        super().__init__()
        self.config = config

        # random 1x1x224x224 image
        self.X = torch.rand(1, 1, 224, 224)
        self.y = 0

        data_module.setup(stage="fit")

        # get a random image from the test set
        self.data_loader = data_module.val_dataloader()

    def plot(self, X, y, pred):
        plt.figure(figsize=(5, 5))
        # rand plot
        X = X.squeeze().numpy()
        X = X[0, :]
        plt.imshow(X)
        plt.axis("off")
        # set tile with prediction and ground truth
        plt.title(f"Prediction: {pred} \n Ground Truth: {y}", fontsize=14)
        plt.tight_layout()
        fig = plt.gcf()
        wandb_im = wandb.Image(fig)
        plt.close()
        return wandb_im

    def tsne_plot(self, eval_features, eval_labels):
        # create a 2-dimensional TSNE model of the images
        tsne = TSNE(n_components=2, random_state=0)
        # fit and transform the image features onto the 2D space
        print(eval_features.shape)
        tsne_obj = tsne.fit_transform(eval_features)
        print(tsne_obj.shape)
        # create a scatter plot with colored labels for the digits
        plt.figure(figsize=(10, 10))
        plt.scatter(
            tsne_obj[:, 0],
            tsne_obj[:, 1],
            c=eval_labels,
            cmap=plt.cm.get_cmap("jet", 4),
            marker="o",
        )
        plt.colorbar(ticks=range(4))
        plt.clim(-0.5, 3.5)
        plt.tight_layout()
        fig = plt.gcf()
        wandb_im = wandb.Image(fig)
        plt.close()
        return wandb_im

    def on_validation_epoch_end(self, trainer, pl_module):
        feature_model = ViTWithoutClassification(pl_module)
        feature_model.eval()

        eval_features = torch.zeros(0, dtype=torch.float32)
        eval_labels = torch.zeros(0, dtype=torch.long)
        for batch in self.data_loader:
            x, y = batch
            x = x.to(device=pl_module.device)
            y = y.to(device=pl_module.device)
            with torch.no_grad():
                x = feature_model(x)
            eval_features = torch.cat((eval_features, x.cpu()), dim=0)
            eval_labels = torch.cat((eval_labels, y.cpu()), dim=0)

        figs = [
            self.tsne_plot(eval_features, eval_labels),
        ]

        # add to logger like so
        trainer.logger.experiment.log({"T-SNE": figs})
