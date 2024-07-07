from moviepy.editor import ImageSequenceClip
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm

from constants import MULTI, SCALE
from practical_rife.model.pytorch_msssim import ssim_matlab
from practical_rife.train_log_RIFE.train_log.RIFE_HDv3 import Model as Model_RIFE
from utils import pad_image

import os

current = os.path.dirname(__file__)

interpolation_model = Model_RIFE()
interpolation_model.load_model(current + "/practical_rife/train_log_RIFE/train_log/", -1)

print("Loaded 3.x/4.x HD model.")

interpolation_model.eval()
interpolation_model.device()


def make_inference(I0, I1, n):
    if interpolation_model.version >= 3.9:
        res = []
        for i in range(n):
            res.append(
                interpolation_model.inference(I0, I1, (i + 1) * 1.0 / (n + 1), SCALE)
            )
        return res
    else:
        middle = interpolation_model.inference(I0, I1, SCALE)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n // 2)
        second_half = make_inference(middle, I1, n=n // 2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]


def interpolate(input_frames, vid_out_path, fps_out=16):
    lastframe = input_frames[0]
    tot_frame = len(input_frames)
    input_frames = input_frames[1:]
    _, h, w = lastframe.shape

    tmp = max(128, int(128 / SCALE))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)

    I1 = lastframe.unsqueeze(0).float()
    I1 = pad_image(I1, padding)
    temp = None  # save lastframe when processing static frame

    result = []
    idx = 0
    while True:
        if temp is not None:
            frame = temp
            temp = None
        elif idx >= tot_frame - 1:
            break
        else:
            frame = input_frames[idx]
            idx += 1

        I0 = I1
        I1 = frame.unsqueeze(0).float()
        I1 = pad_image(I1, padding)

        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)

        ssim = 0.995

        break_flag = False
        if ssim > 0.996:
            if idx >= tot_frame - 1:
                break_flag = True
                frame = lastframe
            else:
                frame = input_frames[idx]
                idx += 1
                temp = frame
            I1 = frame.unsqueeze(0).float()
            I1 = pad_image(I1, padding)
            I1 = interpolation_model.inference(I0, I1, SCALE)
            I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
            frame = I1[0].permute(1, 2, 0)[:h, :w]

        if ssim < 0.2:
            output = []
            for i in range(MULTI - 1):
                output.append(I0)
        else:
            output = make_inference(I0, I1, MULTI - 1)

        lastframe_np = lastframe.cpu().numpy()
        lastframe_np = lastframe_np.astype(np.uint8)
        result.append(lastframe)
        for mid in output:
            result.append(mid[0][:h, :w])
        pbar.update(1)
        lastframe = frame
        if break_flag:
            break

    lastframe_np = lastframe.cpu().numpy()
    lastframe_np = lastframe_np.astype(np.uint8)

    result.append(lastframe)

    clip = ImageSequenceClip(
        [(i.permute(1, 2, 0) * 255).byte().cpu().numpy() for i in result], fps=fps_out
    )
    clip.write_videofile(vid_out_path)

    pbar.close()
    return result
