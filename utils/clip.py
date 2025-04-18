import torch
from transformers import CLIPModel, CLIPProcessor


@torch.no_grad()
def evaluate_prompt_adherence(device, pipeline, loader):
    """
    Evaluates prompt adherence of a generative pipeline using CLIP similarity.

    Args:
        device (torch.device): Device on which to run CLIP model.
        pipeline (DiffusionPipeline): Image generation pipeline.
        loader (DataLoader): DataLoader yielding (images, prompts) batches.

    Returns:
        float: Mean CLIP similarity score measuring prompt adherence.
    """

    if not hasattr(evaluate_prompt_adherence, "_clip_model"):
        evaluate_prompt_adherence._clip_model = (
            CLIPModel.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
        )
        evaluate_prompt_adherence._clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14", use_fast=False
        )

    clip_model = evaluate_prompt_adherence._clip_model
    clip_processor = evaluate_prompt_adherence._clip_processor

    all_scores = []

    for _, prompts in loader:
        generated = pipeline(list(prompts)).images

        inputs = clip_processor(text=list(prompts), images=generated, return_tensors="pt", padding=True).to(device)

        outputs = clip_model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        similarity = (image_embeds * text_embeds).sum(dim=-1)
        all_scores.append(similarity.cpu())

    clip_score = torch.cat(all_scores).mean().item()
    return clip_score
