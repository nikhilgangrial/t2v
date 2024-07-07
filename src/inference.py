from utils import update_path

update_path()

from show1 import generate_base_frames
from psrt import super_resolution
from rife import interpolate
from safa import enhance

def generate_video(prompt_text="panda dancing", negative_prompt="", filename=None):
    filename = filename if filename else prompt_text + ".mp4"

    base_frames = generate_base_frames(prompt_text, negative_prompt, filename="out/base/" + filename)
    super_frames = super_resolution(base_frames, "out/super/" + filename)
    interpolated_frames = interpolate(super_frames, "out/interpol/" + filename)
    enhance(interpolated_frames, "out/enhanced/" + filename)

generate_video()