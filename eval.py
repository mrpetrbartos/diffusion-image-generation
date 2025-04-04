import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from diffusers import DDPMPipeline
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from utils.config import load_config
from utils.dataset import LoadDataset


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class FIDCalculator(pl.LightningModule):
    def __init__(self, pipeline_path: str, dataset_name: str, num_images=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

        self.num_images = num_images
        self.batch_size = batch_size

        self.pipeline = DDPMPipeline.from_pretrained(pipeline_path).to(get_device())

        self.dataset = LoadDataset(dataset_name, split="test", image_size=config["image_size"]).dataset

        self.fid = FrechetInceptionDistance(feature=2048, normalize=True)
        self.transform = T.Compose([T.Resize((299, 299)), T.ToTensor()])

    def get_real_dataloader(self):
        def transform(example):
            example["pixel_values"] = self.transform(example["jpg"])
            return example

        dataset = self.dataset["test"].map(transform)
        dataset.set_format(type="torch", columns=["pixel_values"])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def get_generated_dataloader(self):
        all_images = []
        for _ in tqdm(range(0, self.num_images, self.batch_size), desc="Generating images"):
            generated = self.pipeline(batch_size=self.batch_size).images
            batch = [self.transform(img).unsqueeze(0) for img in generated]
            all_images.extend(batch)

        tensors = torch.cat(all_images, dim=0)[: self.num_images]
        dataset = torch.utils.data.TensorDataset(tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def compute_fid(self):
        self.fid.to(get_device())

        # Add real images
        real_loader = self.get_real_dataloader()
        for batch in tqdm(real_loader, desc="Processing real images"):
            imgs = batch["pixel_values"].to(get_device())
            self.fid.update(imgs, real=True)

        # Add generated images
        gen_loader = self.get_generated_dataloader()
        for batch in tqdm(gen_loader, desc="Processing generated images"):
            imgs = batch[0].to(get_device())
            self.fid.update(imgs, real=False)

        score = self.fid.compute()
        print(f"The number of parameters is {sum(p.numel() for p in self.pipeline.unet.parameters())}")
        print(f"FID Score: {score.item():.4f}")
        return score


if __name__ == "__main__":
    config = load_config("configs/eval_config.yaml")

    model = FIDCalculator(
        pipeline_path=config["model_path"],
        dataset_name="bigdata-pw/TheSimpsons",
        num_images=config["num_images"],
        batch_size=config["batch_size"],
    )
    model.compute_fid()
