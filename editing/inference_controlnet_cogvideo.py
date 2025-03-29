import os
import random
import torch
import numpy as np
from PIL import Image
import argparse
from decord import VideoReader
from diffusers import (
    AutoencoderKLCogVideoX,
    # CogVideoXDDIMScheduler,
)
from diffusers.utils import export_to_video
from invert_scheduler_ddim_cogvideox import CogVideoXDDIMScheduler
from invert_pipeline_control_cogvideo_mask import CogVideoXInvertControlNetPipeline
from pipeline_control_cogvideo import CogVideoXControlNetPipeline
from controlnet import import_controlnet_module
# add dynamic method
import vae_tile
import controlnet_transformer_3d


def load_config(config_path):
    """Load configuration from a Python file."""
    config = {}
    with open(config_path, "r") as f:
        exec(f.read(), config)
    return config


def prepare_video(video_path):
    """Prepare video tensor from the given video path."""
    vr = VideoReader(uri=video_path, height=-1, width=-1)
    ori_vlen = len(vr)
    temp_frms = vr.get_batch(np.arange(0, ori_vlen)).asnumpy()
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    tensor_frms = tensor_frms.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]
    return (tensor_frms - 127.5) / 127.5


def prepare_sketch_input(paths):
    """Prepare sketch input tensors from the given paths."""
    condition_input_list = []
    for path in paths:
        frame = Image.open(path)
        frame = np.array(frame)
        temp_sketch = (frame - 127.5) / 127.5
        if len(temp_sketch.shape) == 2:
            frame_tensor = torch.tensor(temp_sketch).unsqueeze(0).repeat(3, 1, 1)
        else:
            frame_tensor = torch.tensor(temp_sketch)[:, :, :3].permute(2, 0, 1)
        condition_input_list.append(frame_tensor)
    condition_input = torch.stack(condition_input_list, dim=0)  # [T, C, H, W]
    return condition_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]


def main(config_path):
    # Load configuration
    config = load_config(config_path)
    video_path = config["video_path"]
    sketch_paths = config["sketch_paths"]
    sketch_mask_path = config["sketch_mask_path"]
    save_dir = config["save_dir"]
    validation_prompt = config["validation_prompt"]
    control_frame_index = config["control_frame_index"]
    guidance_scale = config["guidance_scale"]
    num_inference_steps = config["num_inference_steps"]
    num_frames = config["num_frames"]
    inversion_fusion = config["inversion_fusion"]

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    front_path = os.path.join(save_dir, "output_")
    if inversion_fusion:
        front_path += "inversion_"

    # Prepare inputs
    tensor_frms = prepare_video(video_path).permute(1, 0, 2, 3).unsqueeze(0)  # [B, C, T, H, W]
    condition_input = prepare_sketch_input(sketch_paths)
    sketch_mask = torch.load(sketch_mask_path)

    # Load models
    vae = AutoencoderKLCogVideoX.from_pretrained(
        config["vae_path"], subfolder="vae", torch_dtype=torch.float16
    )
    vae.enable_tiling()
    vae.requires_grad_(False)

    CogVideoControlNetModel = import_controlnet_module("full")
    controlnet = CogVideoControlNetModel.from_pretrained(
        config["controlnet_path"], torch_dtype=torch.float16, use_safetensors=True
    )
    # whether utilize the inversion fusion
    if inversion_fusion:
        pipeline = CogVideoXInvertControlNetPipeline.from_pretrained(
            config["pipeline_path"], vae=vae, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
        )
    else:
        pipeline = CogVideoXControlNetPipeline.from_pretrained(
            config["pipeline_path"], vae=vae, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
        )

    # Configure pipeline
    device = "cuda"
    pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    # Inference
    if config["num_seeds"] is None:
        seed = config["seed"]
        torch.manual_seed(seed)
        generator = torch.Generator().manual_seed(seed)

        output_path = front_path + f"{seed}_g{guidance_scale}.mp4"
        video = pipeline(
            prompt=validation_prompt,
            image=condition_input,
            video_input=tensor_frms,
            sketch_mask=sketch_mask,
            control_frame_index=control_frame_index,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

        export_to_video(video, output_path, fps=config["fps"])
    else:
        for i in range(config["num_seeds"]):
            print("Inference:", i + 1)
            seed = random.randint(1, 10000)
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)

            output_path = front_path + f"{seed}_g{guidance_scale}.mp4"
            video = pipeline(
                prompt=validation_prompt,
                image=condition_input,
                video_input=tensor_frms,
                sketch_mask=sketch_mask,
                control_frame_index=control_frame_index,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=generator,
            ).frames[0]

            export_to_video(video, output_path, fps=config["fps"])

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference script for CogVideo with ControlNet.")
    # ===============================================================================
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file containing the parameters.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path  # Path to the configuration file
    main(config_path)
