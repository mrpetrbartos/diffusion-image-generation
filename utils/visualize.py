import os

from PIL import Image


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))

    return grid


def generate_samples(save_path, epoch, pipeline):
    prompts = ["A man and a woman in a living room."] * 9

    # Sample some images from random noise (backward diffusion process)
    output = pipeline(prompts, num_inference_steps=50)
    images = output.images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=3, cols=3)

    # Save the images
    os.makedirs(save_path, exist_ok=True)
    image_grid.save(f"{save_path}/{epoch:04d}.png")
