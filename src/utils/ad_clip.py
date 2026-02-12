from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import open_clip


def create_clip_model_and_tokenizer(
    model_name: str,
    pretrained: str,
    device: torch.device,
    allow_download: bool = True,
):
    resolved_pretrained = pretrained
    pretrained_path = Path(pretrained)

    if pretrained_path.exists():
        resolved_pretrained = str(pretrained_path)
    elif not allow_download:
        raise FileNotFoundError(
            f"CLIP pretrained file not found: {pretrained}. "
            "Set a valid local path (e.g. CLIP/ViT-L-14-336px.pt) or enable download in config."
        )

    model, _, _ = open_clip.create_model_and_transforms(
        model_name=model_name,
        pretrained=resolved_pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, tokenizer


def encode_text_with_prompt_ensemble(
    clip_model,
    tokenizer: Callable,
    obj_name: str,
    device: torch.device,
) -> torch.Tensor:
    prompt_normal = [
        "{}",
        "flawless {}",
        "perfect {}",
        "unblemished {}",
        "{} without flaw",
        "{} without defect",
        "{} without damage",
    ]
    prompt_abnormal = ["damaged {}", "broken {}", "{} with flaw", "{} with defect", "{} with damage"]
    prompt_state = [prompt_normal, prompt_abnormal]
    prompt_templates = [
        "a bad photo of a {}.",
        "a low resolution photo of the {}.",
        "a bad photo of the {}.",
        "a cropped photo of the {}.",
        "a bright photo of a {}.",
        "a dark photo of the {}.",
        "a photo of my {}.",
        "a photo of the cool {}.",
        "a close-up photo of a {}.",
        "a black and white photo of the {}.",
        "a bright photo of the {}.",
        "a cropped photo of a {}.",
        "a jpeg corrupted photo of a {}.",
        "a blurry photo of the {}.",
        "a photo of the {}.",
        "a good photo of the {}.",
        "a photo of one {}.",
        "a close-up photo of the {}.",
        "a photo of a {}.",
        "a low resolution photo of a {}.",
        "a photo of a large {}.",
        "a blurry photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a good photo of a {}.",
        "a photo of the small {}.",
        "a photo of the large {}.",
        "a black and white photo of a {}.",
        "a dark photo of a {}.",
        "a photo of a cool {}.",
        "a photo of a small {}.",
        "there is a {} in the scene.",
        "there is the {} in the scene.",
        "this is a {} in the scene.",
        "this is the {} in the scene.",
        "this is one {} in the scene.",
    ]

    text_features = []
    with torch.no_grad():
        for states in prompt_state:
            prompted_state = [state.format(obj_name) for state in states]
            prompted_sentence = []
            for sentence in prompted_state:
                for template in prompt_templates:
                    prompted_sentence.append(template.format(sentence))

            tokens = tokenizer(prompted_sentence).to(device)
            class_embeddings = clip_model.encode_text(tokens)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            text_features.append(class_embedding)

    return torch.stack(text_features, dim=1).to(device)
