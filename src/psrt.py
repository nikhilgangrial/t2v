from moviepy.editor import ImageSequenceClip
import torch
import os

from constants import DEVICE
from rethinkvsralignment.archs.psrt_recurrent_arch import BasicRecurrentSwin
from show_1.showone.pipelines.pipeline_t2v_base_pixel import tensor2vid

device = torch.device(DEVICE)

current = os.path.dirname(__file__)

PSRT_model_path = current + "/rethinkvsralignment/experiments/PSRT_Reccurrent/PSRT_Vimeo.pth"
sypnet_path = current + "/rethinkvsralignment/experiments/pretrained_models/flownet/spynet_sintel_final-3d2a1287.pth"

PSRT_model = BasicRecurrentSwin(
    mid_channels=64,
    embed_dim=120,
    depths=[6, 6, 6],
    num_heads=[6, 6, 6],
    window_size=[3, 8, 8],
    num_frames=3,
    cpu_cache_length=100,
    is_low_res_input=True,
    spynet_path=sypnet_path,
)

PSRT_model.load_state_dict(torch.load(PSRT_model_path)["params"], strict=False)
PSRT_model.eval()
PSRT_model = PSRT_model.to(device)


def super_resolution(frames_tensor, vid_out=None, fps_out=16):
    with torch.no_grad():
        outputs = PSRT_model(frames_tensor.unsqueeze(0))

    if vid_out:
        clip = ImageSequenceClip(
            tensor2vid(outputs.permute(0, 2, 1, 3, 4)), fps=fps_out
        )
        clip.write_videofile(vid_out)

    return list(torch.unbind(outputs.squeeze(0), dim=0))
