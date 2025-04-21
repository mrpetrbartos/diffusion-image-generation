from os import path as ospath
from sys import path

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

path.append(ospath.abspath(".."))

from utils.config import load_config
from utils.dataset import LoadDataset


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class GenerativeEvaluator(pl.LightningModule):
    def __init__(self, pipeline_path: str, dataset_name: str, num_images=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

        self.num_images = num_images
        self.batch_size = batch_size

        self.pipeline = StableDiffusionPipeline.from_pretrained(pipeline_path)
        self.pipeline.to(get_device())
        self.pipeline.set_progress_bar_config(disable=True)

        self.dataset = LoadDataset(dataset_name, split="test", image_size=config["image_size"])

        self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
        self.transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])

    def compute_fid(self):
        self.fid.to(get_device())

        real_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        for batch in tqdm(real_loader, desc="Processing real images"):
            imgs = batch[0].to(get_device())
            self.fid.update(imgs, real=True)

        for i in tqdm(range(0, self.num_images, self.batch_size), desc="Processing generated images"):
            prompts = [self.dataset[j][1] for j in range(i, min(i + self.batch_size, self.num_images))]
            images = self.pipeline(prompt=prompts).images
            imgs = torch.stack([self.transform(img) for img in images]).to(get_device())
            self.fid.update(imgs, real=False)

        score = self.fid.compute()
        print(f"The number of parameters is {sum(p.numel() for p in self.pipeline.unet.parameters())}")
        print(f"FID Score: {score.item():.4f}")
        return score


if __name__ == "__main__":
    config = load_config("../configs/eval_config.yaml")

    model = GenerativeEvaluator(
        pipeline_path=config["model_path"],
        dataset_name="bigdata-pw/TheSimpsons",
        num_images=config["num_images"],
        batch_size=config["batch_size"],
    )

    model.compute_fid()
