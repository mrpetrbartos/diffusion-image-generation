import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append(os.path.abspath(".."))

from diffusers import AutoencoderKL

from models.unet import ConditionalUNet
from utils.clip import evaluate_prompt_adherence
from utils.config import load_config, parse_args
from utils.dataset import LoadDataset
from utils.visualize import generate_samples


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
    train_dataset = LoadDataset(
        "bigdata-pw/TheSimpsons", split="train", image_size=train_cfg["sample_size"], caption_detail=0
    )
    test_dataset = LoadDataset(
        "bigdata-pw/TheSimpsons", split="test", image_size=train_cfg["sample_size"], caption_detail=0
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # Initialize model, autoencoder, CLIP embeddings & noise scheduler
    model = ConditionalUNet(config=model_cfg)
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=scheduler_cfg["num_train_timesteps"],
    )

    # Compile autoencoder for better performance
    vae = torch.compile(vae)

    # Define optimizer, loss function and learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    loss_fn = nn.MSELoss()
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_cfg["lr_warmup_steps"],
        num_training_steps=(len(train_dataloader) * train_cfg["num_epochs"]),
    )

    global_step = 0
    start_epoch = 0
    clip_score = None

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if accelerator.is_main_process:
            print(f"Resuming training from {args.resume}, starting at epoch {start_epoch}")

    model, vae, text_encoder, tokenizer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, vae, text_encoder, tokenizer, optimizer, train_dataloader, lr_scheduler
    )

    # Training loop
    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for _, batch in enumerate(train_dataloader):
            clean_images, captions = batch

            inputs = tokenizer(
                captions, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
            )
            input_ids = inputs.input_ids.to(clean_images.device)
            attention_mask = inputs.attention_mask.to(clean_images.device)

            unwrapped_vae = accelerator.unwrap_model(vae)

            with torch.no_grad():
                latents = unwrapped_vae.encode(clean_images).latent_dist.sample()
                latents = latents * 0.18215  # scaling factor from Stable Diffusion convention
                text_embeddings = text_encoder(input_ids, attention_mask=attention_mask)[0]

            # Conditional training with Classifier-Free Guidance Dropout afterwards
            cfg_prob = train_cfg["cfg_dropout_prob"]
            if cfg_prob > 0:
                # Randomly decide which samples should drop the text embeddings
                drop_mask = torch.rand(text_embeddings.size(0), device=text_embeddings.device) < cfg_prob
                null_embeddings = torch.zeros_like(text_embeddings)
                text_embeddings[drop_mask] = null_embeddings[drop_mask]
                attention_mask[drop_mask] = 0

            # Sample noise to add to the images
            noise = torch.randn(latents.shape, device=clean_images.device)
            bs = latents.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
                dtype=torch.int64,
            )

            # Add noise to the image latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                    encoder_attention_mask=attention_mask,
                )[0]

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
                "clip": clip_score,
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline(
                vae=unwrapped_vae,
                text_encoder=accelerator.unwrap_model(text_encoder),
                tokenizer=tokenizer,
                unet=accelerator.unwrap_model(model).unet,
                scheduler=noise_scheduler,
                safety_checker=None,
                feature_extractor=None,
            )

            if (n := train_cfg["eval_every_n_epochs"]) > 0 and (epoch + 1) % n == 0:
                clip_score = evaluate_prompt_adherence(accelerator.device, pipeline, test_loader)

            # Infer images from random noise and save them for reference
            if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0 or epoch == train_cfg["num_epochs"] - 1:
                generate_samples(train_cfg["save_image_path"], epoch, pipeline)

            # Save the model
            if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0 or epoch == train_cfg["num_epochs"] - 1:
                pipeline.save_pretrained(train_cfg["save_model_path"])
                torch.save(
                    {
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    f"chckpt_epoch_{epoch}.pth",
                )


if __name__ == "__main__":
    train()
