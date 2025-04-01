import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from models.unet import UNet
from utils.config import load_config, parse_args
from utils.dataset import LoadDataset
from utils.evaluate import evaluate


def train():
    # Parse arguments
    args = parse_args()

    # Load config
    config = load_config(args.config)
    train_cfg = config["training"]
    model_cfg = config["model"]
    scheduler_cfg = config["scheduler"]

    accelerator = Accelerator(
        mixed_precision=train_cfg["mixed_precision"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
    )

    # Load dataset
    dataset = LoadDataset("nlphuji/flickr30k", split="test", image_size=model_cfg["sample_size"])
    train_dataloader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )

    # Initialize model & noise scheduler
    model = UNet(config=model_cfg)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=scheduler_cfg["num_train_timesteps"],
    )

    # Define optimizer, loss function and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    loss_fn = nn.MSELoss()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_cfg["lr_warmup_steps"],
        num_training_steps=(len(train_dataloader) * train_cfg["num_epochs"]),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Training loop
    for epoch in range(train_cfg["num_epochs"]):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for _, batch in enumerate(train_dataloader):
            clean_images = batch

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
                dtype=torch.int64,
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps)[0]
                loss = loss_fn(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            # Infer images from random noise and save them for reference
            if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0 or epoch == train_cfg["num_epochs"] - 1:
                evaluate(train_cfg["save_image_path"], epoch, pipeline, train_cfg["batch_size"])

            # Save the model
            if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0 or epoch == train_cfg["num_epochs"] - 1:
                pipeline.save_pretrained(train_cfg["save_model_path"])
                torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
