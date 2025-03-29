import os
import torch
import numpy as np
import argparse
from PIL import Image

from diffusers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video

from pipeline_control_cogvideo import CogVideoXControlNetPipeline
from controlnet import import_controlnet_module
# import package to add dynamic method
import controlnet_transformer_3d
import vae_tile

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Inference script for CogVideo with ControlNet.")
    # ===============================================================================
    parser.add_argument(
        "--text_path",
        type=str,
        required=True,
        help="Path to the input text file containing the prompt.",
    )
    parser.add_argument(
        "--image_paths",
        type=str,
        required=True,
        help="Comma-separated paths to one or two input images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output results.",
    )
    parser.add_argument(
        "--controlnet_name",
        type=str,
        required=True,
        help="Name of the ControlNet model to use.",
    )
    parser.add_argument(
        "--control_frame_index",
        type=str,
        required=True,
        help="Control frame index, e.g., '0' for one image or '0,12' for two images.",
    )
    parser.add_argument(
        "--control_checkpoint_path",
        type=str,
        default="",
        help="The Path of the controlnet checkpoint",
    )
    parser.add_argument(
        "--cogvideo_checkpoint_path",
        type=str,
        default="",
        help="The Path of the cogvideo checkpoint",
    )
    parser.add_argument(
        "--control_scale",
        type=float,
        default=1.0,
        help="Control scale value.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10.0,
        help="Guidance scale value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

def parse_control_frame_index(input_str):
    try:
        return list(map(int, input_str.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Control frame index must be a comma-separated list of integers, e.g., '0' or '0,12'.")

def load_images(image_paths):
    condition_input_list = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image_array = np.array(image)
        normalized_image = (image_array - 127.5) / 127.5

        if len(normalized_image.shape) == 2:
            tensor_image = torch.tensor(normalized_image).unsqueeze(0).repeat(3, 1, 1)
        else:
            tensor_image = torch.tensor(normalized_image).permute(2, 0, 1)[:3]

        condition_input_list.append(tensor_image)

    condition_input = torch.stack(condition_input_list, dim=0)  # [T, C, H, W]
    condition_input = condition_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
    return condition_input

if __name__ == "__main__":
    args = parse_args()

    # Parse image paths and control frame index
    image_paths = args.image_paths.split(',')
    control_frame_index = parse_control_frame_index(args.control_frame_index)
    if len(image_paths) != len(control_frame_index):
        raise ValueError("Number of images must match the number of control frame indices.")
    # cfg requires two control frames index
    control_frame_index = [control_frame_index, control_frame_index]

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Load ControlNet model
    CogVideoControlNetModel = import_controlnet_module(args.controlnet_name)
    controlnet = CogVideoControlNetModel.from_pretrained(
        args.control_checkpoint_path, torch_dtype=torch.float16, use_safetensors=True
    )
    print("Control block index:", controlnet.control_block_index)

    # Load pipeline
    pipeline = CogVideoXControlNetPipeline.from_pretrained(
        args.cogvideo_checkpoint_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to('cuda')
    pipeline.vae.enable_tiling()

    # Load images and text
    condition_input = load_images(image_paths)
    print("Condition input shape:", condition_input.shape)

    with open(args.text_path, "r", encoding='utf-8') as f:
        validation_prompt = f.read()

    # Set seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
    else:
        generator = torch.Generator()

    # Generate video
    output_path = os.path.join(
        args.output_dir,
        f"{os.path.basename(args.text_path)[:-4]}_seed{args.seed}_g{args.guidance_scale}_c{args.control_scale}.mp4"
    )
    video = pipeline(
        prompt=validation_prompt,
        image=condition_input,
        control_frame_index=control_frame_index,
        num_videos_per_prompt=1,
        num_inference_steps=50,
        num_frames=49,
        use_dynamic_cfg=True,
        guidance_scale=args.guidance_scale,
        generator=generator,
        controlnet_conditioning_scale=args.control_scale,
    ).frames[0]

    export_to_video(video, output_path, fps=8)
    print(f"Video saved to {output_path}")

