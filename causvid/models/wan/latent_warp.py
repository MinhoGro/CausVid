import torch, torch.nn.functional as F
import numpy as np
import math
import matplotlib
import imageio
import logging

class LatentWarper:
    def __init__(self, pre_block, cur_block, *args, **kwargs):
        self.pre_block = pre_block
        self.cur_block = cur_block

    def flow_frame_visualize(self, flow):
        dx, dy = torch.chunk(flow, chunks=2, dim=-1)
        angle = torch.atan2(dx, dy)
        hue = (angle + math.pi) / (2 * math.pi)
        mag = torch.sqrt(dx * dx + dy * dy)
        mag_norm = (mag / mag.max().clamp(min=1e-6)).clamp(0, 1)

        sat = 0.9
        hsv = np.stack([hue, sat * np.ones_like(hue), mag_norm], axis=-1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return rgb

    def view_flow(self, flow):
        rgb_flow = []
        for f in flow:
            rgb_flow.append(self.flow_frame_visualize(f))

        imageio.mimsave(f"flow_video.mp4", rgb_flow, fps=16, codec="libx264")

    def token_flow(self, t, h_patch, w_patch):
        # t = F.normalize(t, dim=-1) # normalize
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

        # visualization, forbidden for now.
        # self.view_flow(flow)
        return flow

    def warp(self):
        # calculate flow
        latent = torch.cat([self.pre_block, self.cur_block], dim=0)     # [2, 3, 16, 60, 104],
                                                                        # denoise_pred size: [1, 3, 16, 60, 104]
        latent = latent.view(-1, 16, 60, 104)
        latent = latent.permute(0, 2, 3, 1)[1:5]  # [6, 60, 104, 16] -> [4, 60, 104, 16]
        latent = latent.view(4, 60*104, 16)
        latent = latent[..., 5]                     # select 5th. channel latent.

        if latent.dim() == 2:
            latent = latent.unsqueeze(-1)
        elif latent.dim() != 3:
            logging.error(f"latent: {latent.shape}, size error! should be 3")

        flow = self.token_flow(latent, 60, 104)  # size: [3, 60, 104, 2]
        flow = torch.stack(flow, dim=0)
        flow = flow.view(3, 60, 104, 2)

        # warp latent
        H, W = self.cur_block.shape[-2:]
        ys = torch.arange(H)
        xs = torch.arange(W)
        coords_y, coords_x = torch.meshgrid(ys, xs, indexing='ij')  # grid_y/grid_x: [H, W],
                                                                    # (grid_y[i, j], grid_x[i, j]) == (i, j)
        coords_y = coords_y.clone().to(self.cur_block.device)
        coords_x = coords_x.clone().to(self.cur_block.device)
        warpt_latent = []
        for index in range(3):
            # 1st layer
            coords_x -= flow[index, ..., 1]
            coords_y -= flow[index, ..., 0]

            coords_x_norm = (coords_x / (W - 1)) * 2 - 1  # [H, W]
            coords_y_norm = (coords_y / (H - 1)) * 2 - 1  # [H, W]
            coords = torch.stack([coords_y_norm, coords_x_norm], dim=-1)  # [H, W, 2]

            new_latent = F.grid_sample(
                self.pre_block[0][2].unsqueeze(0).float(),
                coords.unsqueeze(0).float(),
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            warpt_latent.append(new_latent)

        return torch.cat(warpt_latent, dim=0).unsqueeze(0)


if __name__ == "__main__":
    name = "../../../latent.pt"
    t = torch.load(name, map_location=torch.device("cpu")) # [B, L, dim]

    pre_block = t[0][0]
    cur_block = t[0][1]

    W = LatentWarper(pre_block, cur_block)
    warpt = W.warp()
    print(warpt.shape)