import argparse
import os

import torch
from diffusers import StableDiffusionPipeline


def infer(args):
    # Set seed for reproducibility
    generator = torch.manual_seed(args.seed)

    # Load the pipeline
    print(f"Loading model from {args.model_path}...")
    pipe = StableDiffusionPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe.to("cuda:7")

    # Inference
    print(f"Generating image for prompt: '{args.prompt}'")
    output = pipe(
        [args.prompt] * args.num_images,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    for i, img in enumerate(output):
        path = args.output_path
        if args.num_images > 1:
            root, ext = os.path.splitext(args.output_path)
            path = f"{root}_{i}{ext}"
        print(f"Saved image to {path}")
        img.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an image from a text prompt using a pretrained diffusion model."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model or model repo id")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt to generate image from")
    parser.add_argument("--output_path", type=str, default="output.png", help="Path to save the generated image")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate")

    args = parser.parse_args()

    infer(args)
