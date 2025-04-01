import os

from PIL import Image


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))

    return grid


def evaluate(save_path, epoch, pipeline, batch_size):
    # Sample some images from random noise (backward diffusion process)
    images = pipeline(batch_size=batch_size).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    os.makedirs(save_path, exist_ok=True)
    image_grid.save(f"{save_path}/{epoch:04d}.png")
