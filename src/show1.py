from moviepy.editor import ImageSequenceClip
import torch

from show_1.showone.pipelines.pipeline_t2v_base_pixel import tensor2vid

from constants import SEED
from show_1.showone.pipelines import TextToVideoIFPipeline

pretrained_model_path = "showlab/show-1-base"
base_model = TextToVideoIFPipeline.from_pretrained(
    pretrained_model_path, torch_dtype=torch.float16, variant="fp16"
)
base_model.enable_model_cpu_offload()


def generate_base_frames(
    prompt_text,
    negative_prompt,
    num_frames=8,
    height=64,
    width=64,
    num_steps=50,
    filename=None,
):
    prompt_embeds, negative_embeds = base_model.encode_prompt(
        prompt_text, negative_prompt
    )

    frames = base_model(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        guidance_scale=9.0,
        generator=torch.manual_seed(SEED),
        output_type="pt",
    ).frames

    if filename:
        clip = ImageSequenceClip(tensor2vid(frames), fps=16)
        clip.write_videofile(filename)

    return frames
