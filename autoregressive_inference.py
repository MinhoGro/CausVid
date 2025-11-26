import logging

from causvid.models.wan.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

pipeline = InferencePipeline(config, device="cuda")
pipeline.to(device="cuda", dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset(args.prompt_file_path)

sampled_noise = torch.randn(
    [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
)

os.makedirs(args.output_folder, exist_ok=True)

for prompt_index in tqdm(range(len(dataset))):
    prompts = [dataset[prompt_index]]

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts
    )[0].permute(0, 2, 3, 1).cpu().numpy()

    if pipeline.generator_model_name == "causal_wan":
        m = pipeline.generator.model
        captured = getattr(m, "captured", None)

        if captured:
            self_attn_buf = captured.get("self_attn", [])
            self_attn = torch.cat(self_attn_buf, dim=1).cpu()  # [B, n*L, dim]
            torch.save(self_attn, "self_attn_tokens.pt")
            self_attn_buf.clear()

            cross_attn_buf = captured.get("cross_attn", [])
            cross_attn = torch.cat(cross_attn_buf, dim=1).cpu()
            torch.save(cross_attn, "cross_attn_tokens.pt")
            cross_attn_buf.clear()

        else:
            logging.error(f'captured tensor is empty!')

        latent_cap = getattr(pipeline, "captured", None)
        if latent_cap:
            # latent [1, 3, 16, 60, 104]
            latent_buf = latent_cap.get("latent", [])
            latent = torch.cat(latent_buf, dim=1).cpu()
            torch.save(latent_buf, "latent.pt")
            latent_buf.clear()

            denoise_latent_buf = latent_cap.get("noise", [])
            denoise_latent = torch.cat(denoise_latent_buf, dim=1).cpu()
            torch.save(denoise_latent, "noise.pt")
            denoise_latent_buf.clear()

    export_to_video(
        video, os.path.join(args.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)
