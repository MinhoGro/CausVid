import torch, torch.nn.functional as F
import numpy as np
import math
import matplotlib
import imageio
import imageio.v3 as iio
import logging
import os
from pathlib import Path

class LatentWarper:
    def __init__(self, pre_block, cur_block, *args, **kwargs):
        self.pre_block = pre_block
        self.cur_block = cur_block
        self.output_dir = '/root/autodl-tmp/CausVid/flows'
        self.localout_dir = '../../../flows'
        self.p = Path(self.output_dir)

    # input:
    #   latent frame, [1,dim,H,W]
    #   flow frame, [H,W,2]
    def _warp_latent(self, latent, flow):
        pre_latent = latent[0].permute(1,2,0) # [H,W,dim]
        new_latent = torch.zeros_like(pre_latent)
        # simple way, 2 for loop, check new latent coordinate
        for h in range(new_latent.shape[0]):
            for w in range(new_latent.shape[1]):
                h_flow = h - flow[h][w][0]
                w_flow = w - flow[h][w][1]
                if h_flow < 0 or w_flow < 0:
                    continue
                elif h_flow > new_latent.shape[0]-1 or w_flow > new_latent.shape[1]-1:
                    continue
                pre = pre_latent[h_flow][w_flow]
                new_latent[h][w] = pre_latent[h_flow][w_flow]

        return new_latent.permute(2,0,1).unsqueeze(0) # return [1,dim=16,H=60,W=104]


    def _has_file(self, b: str) -> bool:
        if os.path.isdir(self.p):
            os.makedirs(self.p, exist_ok=True)
        else:
            self.p = Path(self.localout_dir)
        # 目录存在且其中有名为 t 的文件则返回 True
        return (self.p / b).is_file()

    def save_flow_video(self, rgb_flow, video_name):
        frames_uint8 = []
        for i, frame in enumerate(rgb_flow):  # frame shape [H,W,3], float in [0,1]
            # frame = [f[..., :]/20 for f in frame]
            vmin, vmax = frame.min(), frame.max()
            denom = (vmax - vmin) + 1e-8
            frame_norm = (frame - vmin) / denom
            img_uint8 = (frame_norm * 255).astype(np.uint8).squeeze(axis=2)
            # if not os.path.isdir(f'{output_dir}'):
            #     os.makedirs(f'{output_dir}', exist_ok=True)
            # imageio.imwrite(f"{output_dir}/frame_{i:03d}.png", img_uint8)
            frames_uint8.append(img_uint8)

        imageio.mimsave(f"{self.p}/{video_name}", frames_uint8, fps=1, codec="libx264")

    def flow_view(self, f):
        dx, dy = torch.chunk(f, chunks=2, dim=-1)
        angle = torch.atan2(dx, dy)
        hue = (angle + math.pi) / (2 * math.pi)
        mag = torch.sqrt(dx * dx + dy * dy)
        mag_norm = (mag / mag.max().clamp(min=1e-6)).clamp(0, 1)

        sat = 0.9
        # sat_t = torch.as_tensor(sat, device=hue.device, dtype=hue.dtype)
        hsv = torch.stack([hue, sat * torch.ones_like(hue), mag_norm], dim=-1)
        rgb = matplotlib.colors.hsv_to_rgb(hsv.cpu().numpy())
        return rgb

    def flow_monitor(self, flow):
        rgb_flow = []
        for f in flow:
            rgb_flow.append(self.flow_view(f))
        block = 0
        while self._has_file(b=f'flow_video{block}.mp4'):
            block += 1
        self.save_flow_video(rgb_flow, video_name=f'flow_video{block}.mp4')

    def token_flow(self, t, h_patch, w_patch):
        # t = F.normalize(t, dim=-1) # normalize
        flow = []
        for index in range(t.shape[0]-1):
            q, k = t[index], t[index+1]
            sim = torch.matmul(q, k.transpose(0, 1))
            idx = sim.argmax(dim=-1)
            map_h = idx // w_patch
            map_w = idx % w_patch
            # map_h, map_w 形状都是 [B, L]，对应每个当前帧 token 在下一帧的 (h, w) 位置

            coords = torch.stack([map_h, map_w], dim=-1)  # [B, L, 2]
            coords = coords.view(h_patch, w_patch, 2)

            ys = torch.arange(h_patch, device=coords.device)
            xs = torch.arange(w_patch, device=coords.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W], [H, W]
            base = torch.stack([grid_y, grid_x], dim=-1)  # [H, W, 2]
            flow_delta = base - coords # [H, W, 2], (Δh, Δw)

            flow.append(flow_delta)

        self.flow_monitor(flow)
        return flow

    def _latent_norm(self, x):
        x_ch_first = x.transpose(1, 2)  # [4, 16, 60*104]

        mn = x_ch_first.amin(dim=-1, keepdim=True)
        mx = x_ch_first.amax(dim=-1, keepdim=True)
        x_norm = (x_ch_first - mn) / (mx - mn + 1e-8)
        latent = x_norm.transpose(1, 2)  # [4, 60*104, 16]
        return latent

    def warp(self):
        # calculate flow
        latent = torch.cat([self.pre_block, self.cur_block], dim=0)     # [2, 3, 16, 60, 104],
                                                                        # denoise_pred size: [1, 3, 16, 60, 104]
        dim = latent.shape[2] # 16
        h_patch, w_patch = latent.shape[-2:] # for latent tensor, size is h=60,w=104
        latent = latent.view(-1, dim, h_patch, w_patch)
        latent = latent.permute(0, 2, 3, 1)[2:6]  # [6, 60, 104, 16] -> [4, 60, 104, 16]
        latent = latent.view(4, h_patch * w_patch, dim)
        latent  = self._latent_norm(latent)
        # latent = latent[..., [5]]                     # select 5th. channel latent.

        if latent.dim() == 2:
            latent = latent.unsqueeze(-1)
        elif latent.dim() != 3:
            logging.error(f"latent: {latent.shape}, size error! should be 3")

        flow = self.token_flow(latent, h_patch, w_patch)  # size: [3, 60, 104, 2]
        flow = torch.stack(flow, dim=0)
        flow = flow.view(3, h_patch, w_patch, 2)

        # view flow distant
        flow_dist = []
        for f in flow:
            f_dist = torch.zeros(f.shape[0], f.shape[1]).to(latent.device)
            for i in range(f.shape[0]):
                for j in range(f.shape[1]):
                    x = f[i][j][0]
                    y = f[i][j][1]
                    f_dist[i][j] = y +x
            flow_dist.append(f_dist)

        # warp latent
        H, W = self.cur_block.shape[-2:]
        ys = torch.arange(H)
        xs = torch.arange(W)
        coords_y0, coords_x0 = torch.meshgrid(ys, xs, indexing='ij')  # grid_y/grid_x: [H, W],
                                                                    # (grid_y[i, j], grid_x[i, j]) == (i, j)
        new_latent = self.pre_block[0][2].unsqueeze(0)
        warpt_latent = []
        for index in range(3):
            # coords_y = coords_y0.clone().to(self.cur_block.device)
            # coords_x = coords_x0.clone().to(self.cur_block.device)
            # coords_x -= flow[index, ..., 1]
            # coords_y -= flow[index, ..., 0]
            #
            # coords_x_norm = (coords_x / (W - 1)) * 2 - 1  # [H, W]
            # coords_y_norm = (coords_y / (H - 1)) * 2 - 1  # [H, W]
            # coords = torch.stack([coords_y_norm, coords_x_norm], dim=-1)  # [H, W, 2]
            #
            # new_latent = F.grid_sample(
            #     new_latent.float(),
            #     coords.unsqueeze(0).float(),
            #     mode='bilinear',
            #     padding_mode='zeros',
            #     align_corners=True
            # )
            # warpt_latent.append(new_latent)

            new_latent = self._warp_latent(new_latent, flow[index])
            warpt_latent.append(new_latent)

        warpt_latent = torch.cat(warpt_latent, dim=0)
        return warpt_latent.unsqueeze(0)

def read_video(video_path, device="cpu"):
    video = iio.imread(video_path)
    video = torch.from_numpy(video).to(device=device, dtype=torch.float32)  # [F, H, W, 3]
    video = video / 255.0
    return video.permute(0, 3, 1, 2) # [F, 3, H, W]

if __name__ == "__main__":
    tensor_name = "../../../latent.pt"
    video_path = "../../../outputs/output_clean.mp4"
    r_video = True
    show_warpt = False

    pre_block, cur_block = None, None
    if r_video:
        video = read_video(video_path)
        pre_block = video[0:3].unsqueeze(0)[..., 210:270, 312:416] # [480, 832] -> [60, 104]
        cur_block = video[3:6].unsqueeze(0)[..., 210:270, 312:416]
        show_warpt = True
    else:
        t = torch.load(tensor_name, map_location=torch.device("cpu")) # [B, L, dim]
        pre_block = t[0][2]
        cur_block = t[0][3]

    W = LatentWarper(pre_block, cur_block)
    warpt = W.warp()
    print(warpt.shape)

    pre_block = pre_block.float()[0].permute(0, 2, 3, 1)[..., [0,1,2]]
    cur_block = cur_block.float()[0].permute(0, 2, 3, 1)[..., [0,1,2]]

    A = pre_block.unsqueeze(-2).cpu().numpy()
    B_ = cur_block.unsqueeze(-2).cpu().numpy()
    W.save_flow_video(A, video_name=f'flow_video_A.mp4')
    W.save_flow_video(B_, video_name=f'flow_video_B_.mp4')

    # when warpt dim=3, save and view warpt
    if show_warpt:
        warpt = warpt.float()[0].permute(0, 2, 3, 1)[..., [0,1,2]] # size = [3,60,104,3]
        B = warpt.unsqueeze(-2).cpu().numpy()
        W.save_flow_video(B, video_name=f'flow_video_W.mp4')