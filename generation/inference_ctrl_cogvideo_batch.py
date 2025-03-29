import os
import torch
import numpy as np
import argparse
import glob
import random
from PIL import Image

from diffusers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video

from pipeline_control_cogvideo import CogVideoXControlNetPipeline
from controlnet import import_controlnet_module
# import package to add dynamic method
import controlnet_transformer_3d
import vae_tile

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    # ===============================================================================
    # Add ablation configs
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="The directory that contains sketch and text",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory of output results",
    )
    parser.add_argument(
        "--controlnet_name",
        type=str,
        required=True,
        help="choose which controlnet is used",
    )
    parser.add_argument(
        "--control_frame_index",
        type=str,
        required=True,
        help="The index of control frames, e.g., '0' for one image or '0,12' for two images.",
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
        help="control_scale value.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=10.0,
        help="guidance_scale value.",
    )
    parser.add_argument(
        "--num_example",
        type=int,
        default=1,
        help="number of examples each input predict.",
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

def validate_inputs(sketch_list, txt_list, control_frame_index):
    """
    Validate the input data based on the control_frame_index.
    """
    expected_ratio = len(control_frame_index)
    if len(sketch_list) != len(txt_list) * expected_ratio:
        raise ValueError(
            f"Mismatch between the number of sketches and text files. "
            f"Expected sketches to be {expected_ratio} times the number of text files, "
            f"but got {len(sketch_list)} sketches and {len(txt_list)} text files."
        )

if __name__ == "__main__":
    args = parse_args()
    
    # 1. Read the sketch and txt list
    sketch_folder_jpg = os.path.join(args.input_dir, '*.jpg')
    sketch_folder_png = os.path.join(args.input_dir, '*.png')
    sketch_list = sorted(glob.glob(sketch_folder_jpg) + glob.glob(sketch_folder_png))
    txt_folder = os.path.join(args.input_dir, '*.txt')
    txt_list = sorted(glob.glob(txt_folder))
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    # 2. Parse and validate control_frame_index
    control_frame_index = parse_control_frame_index(args.control_frame_index)
    print("control_frame_index:", control_frame_index)
    
    # Validate inputs
    validate_inputs(sketch_list, txt_list, control_frame_index)
    
    # Adjust sketch_list structure based on control_frame_index
    def split_list_into_chunks(original_list, chunk_size):
        return [original_list[i:i + chunk_size] for i in range(0, len(original_list), chunk_size)]
    
    # Example:
    # two keyframes input: ['ex1_1', 'ex1_2', 'ex2_1', 'ex2_2'] -> [['ex1_1', 'ex1_2'], ['ex2_1', 'ex2_2']]
    # single keyframe input: ['ex1_1', 'ex1_2', 'ex2_1', 'ex2_2'] -> [['ex1_1'], ['ex1_2'], ['ex2_1'], ['ex2_2']]
    sketch_list = split_list_into_chunks(sketch_list, len(control_frame_index))
    control_frame_index = [control_frame_index, control_frame_index]

    # 3. Prepare the CogVideo model
    CogVideoControlNetModel = import_controlnet_module(args.controlnet_name)
    controlnet = CogVideoControlNetModel.from_pretrained(
        args.control_checkpoint_path, torch_dtype=torch.float16, use_safetensors=True
    )
    print("control_block_index:", controlnet.control_block_index)
    
    pipeline = CogVideoXControlNetPipeline.from_pretrained(
        args.cogvideo_checkpoint_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    device = 'cuda'
    pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.vae.enable_tiling()
    
    # 4. Process each example
    for ex_id in range(len(txt_list)):
        # 1. Read the video name
        ex_name = txt_list[ex_id].split('/')[-1]
        print("process:", ex_id, "/", len(txt_list))
        print(ex_name)
        
        # 2. Read the sketch images
        key_frame_list = sketch_list[ex_id]
        condition_input_list = []
        for key_frame_path in key_frame_list:
            begin_frame_canny = Image.open(key_frame_path)
            begin_frame_canny = np.array(begin_frame_canny)
            temp_sketch = (begin_frame_canny - 127.5) / 127.5

            if len(temp_sketch.shape) == 2:
                begin_frame_canny_tensor = torch.tensor(temp_sketch).unsqueeze(0)
                begin_frame_canny_tensor = begin_frame_canny_tensor.repeat(3,1,1)
            else:
                begin_frame_canny_tensor = torch.tensor(temp_sketch)
                print("sketch shape:", begin_frame_canny_tensor.shape)
                begin_frame_canny_tensor = begin_frame_canny_tensor[:,:,0:3].permute(2,0,1)

            condition_input_list.append(begin_frame_canny_tensor)

        condition_input = torch.stack(condition_input_list, dim=0)  # [T, C, H, W]
        condition_input = condition_input.unsqueeze(0)  # [B, T, C, H, W]
        condition_input = condition_input.permute(0, 2, 1, 3, 4) # [B, C, T, H, W]
        print("condition_input:", condition_input.shape)
        
        # 3. Read the caption
        caption_path = txt_list[ex_id]
        with open(caption_path, "r", encoding='utf-8') as f:
            validation_prompt = f.read()
        
        # 4. Predict the video results
        guidance_scale = args.guidance_scale
        control_scale = args.control_scale
        
        # 5. Inference the video results
        for i in range(args.num_example):
            seed = random.randint(1,10000)
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            front_path = os.path.join(args.output_dir, ex_name[:-4])
            back_path = "_seed" + str(seed) + "_g" + str(guidance_scale) + "_c" + str(control_scale) + ".mp4"
            output_path = front_path + back_path
            
            video = pipeline(
                prompt=validation_prompt,
                image=condition_input, # Control input images
                control_frame_index=control_frame_index,
                num_videos_per_prompt=1,  # Number of videos to generate per prompt
                num_inference_steps=50,  # Number of inference steps
                num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.31.0` and after.
                use_dynamic_cfg=True,  ## This id used for DPM Sechduler, for DDIM scheduler, it should be False
                guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance, can set to 7 for DPM scheduler
                generator=generator,  # Set the seed for reproducibility
                controlnet_conditioning_scale=control_scale,
            ).frames[0]

            export_to_video(video, output_path, fps=8)

