import os

import torch
import yaml
import torch.nn.functional as F
import math
import matplotlib
import imageio, numpy as np

# patch = (1, 2, 2)
# latent -> token, [H/2, W/2] = 30 * 52
# frame per block -> para.num_frame_per_block

def flow_view(flow):
    dx, dy = torch.chunk(flow, chunks=2, dim=-1)
    angle = torch.atan2(dx, dy)
    hue = (angle + math.pi) / (2 * math.pi)
    mag = torch.sqrt(dx * dx + dy * dy)
    mag_norm = (mag / mag.max().clamp(min=1e-6)).clamp(0, 1)

    sat = 0.9
    hsv = np.stack([hue, sat * np.ones_like(hue), mag_norm], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb

def token_flow(t, h_patch, w_patch):
    t = F.normalize(t, dim=-1) # normalize
    flow = []
    for q, k in zip(t[:-1], t[1:]):
        sim = torch.matmul(q, k.transpose(0, 1))
        idx = sim.argmax(dim=-1)
        map_h = idx // w_patch
        map_w = idx % w_patch
        # map_h, map_w 形状都是 [B, L]，对应每个当前帧 token 在下一帧的 (h, w) 位置

        coords = torch.stack([map_h, map_w], dim=-1)  # [B, L, 2]
        coords = coords.view(h_patch, w_patch, 2)
        flow.append(coords)
    return flow

def main():
    name = "denoise_latent.pt"
    t = torch.load(name, map_location=torch.device("cpu")) # [B, L, dim]
    # torch.Size([1, 131040, 1536])
    # latent size [60, 104]
    # token size [L, 30, 52]
    if name == "denoise_latent.pt":
        for i, _t in enumerate(t):
            if _t.shape[0] == 1 :
                t[i] = _t[0]
    else:
        print(t.shape)

    token_shape = []
    with open("configs/wan_causal_dmd.yaml", "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        token_shape = cfg['image_or_video_shape']

    dim = t.shape[-1]
    h_patch = token_shape[3] // 2
    w_patch = token_shape[4] // 2
    n_frames = t.shape[1] // (h_patch * w_patch)

    if name == "latent.pt" or name == "denoise_latent.pt":
        t = t.permute(1, 2, 3, 0)
        print(t.shape)
        t = t.view(t.shape[0], h_patch * w_patch * 4, t.shape[-1])
        # latent size: [16, 84, 60, 104], dim = 16, frames = 84
        flow = token_flow(t, h_patch *2, w_patch *2)
    else:
        t = t.view(1, n_frames, h_patch * w_patch, dim)
        print(t.shape)

        flow = token_flow(t[0], h_patch, w_patch)
    rgb_flow = []
    for f in flow:
        rgb_flow.append(flow_view(f))

    frames_uint8 = []
    for i, frame in enumerate(rgb_flow):  # frame shape [H,W,3], float in [0,1]
        img_uint8 = (np.clip(frame, 0, 1) * 255).astype(np.uint8).squeeze(axis=2)
        output_dir = 'flow_images'
        if not os.path.isdir(f'{output_dir}'):
            os.makedirs(f'{output_dir}', exist_ok=True)
        imageio.imwrite(f"{output_dir}/frame_{i:03d}.png", img_uint8)
        frames_uint8.append(img_uint8)

    imageio.mimsave("flow_video.mp4", frames_uint8, fps=16, codec="libx264")

if __name__ == '__main__':
    main()