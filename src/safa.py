from moviepy.editor import ImageSequenceClip
from torch.nn import functional as F
from tqdm import tqdm

from practical_rife.train_log_SAFA.train_log.model import Model as Model_SAFA

from practical_rife.model.pytorch_msssim import ssim_matlab
from utils import pad_image
import os

current = os.path.dirname(__file__)

enhancer_model = Model_SAFA()
enhancer_model.load_model(current + "/practical_rife/train_log_SAFA/train_log/")

print("Loaded SAFA model.")

enhancer_model.eval()
enhancer_model.device()


def enhance(input_frames, vid_out_path, fps_out=16):
    lastframe = input_frames[0]
    tot_frame = len(input_frames)

    input_frames = input_frames[1:]
    _, h, w = lastframe.shape
    print(lastframe.shape)

    tmp = 64
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    padding = (0, pw - w, 0, ph - h)
    pbar = tqdm(total=tot_frame)

    enhanced_frames = []
    idx = 0
    while idx < tot_frame - 1:
        frame = input_frames[idx]
        idx += 1

        I0 = pad_image(lastframe.unsqueeze(0).float(), padding, sr=True)
        I1 = pad_image(frame.unsqueeze(0).float(), padding, sr=True)
        I0_small = F.interpolate(I0, (32, 32), mode="bilinear", align_corners=False)
        I1_small = F.interpolate(I1, (32, 32), mode="bilinear", align_corners=False)

        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        if ssim < 0.2:
            out = [
                enhancer_model.inference(I0, I0, [0])[0],
                enhancer_model.inference(I1, I1, [0])[0],
            ]
        else:
            out = enhancer_model.inference(I0, I1, [0, 1])
        assert len(out) == 2

        enhanced_frames.append(out[0].squeeze(0))
        enhanced_frames.append(out[1].squeeze(0))

        if idx >= tot_frame - 1:
            break
        lastframe = input_frames[idx]
        idx += 1
        pbar.update(2)

    clip = ImageSequenceClip(
        [(i.permute(1, 2, 0) * 255).byte().cpu().numpy() for i in enhanced_frames],
        fps=fps_out,
    )
    clip.write_videofile(vid_out_path)
    pbar.close()
    return enhanced_frames
