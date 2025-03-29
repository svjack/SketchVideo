import os
import json
import argparse
import cv2
import numpy as np
import torch
from decord import VideoReader
import imageio

def load_config(config_path):
    """Load configuration from a Python file."""
    config = {}
    with open(config_path, "r") as f:
        exec(f.read(), config)
    return config

def dense_optical_flow(method, old_frame, new_frame, params=[], to_gray=False):
    """Calculate dense optical flow between two frames."""
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_frame = cv2.resize(old_frame, dsize=(360, 240))
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame = cv2.resize(new_frame, dsize=(360, 240))
    flow = method(old_frame, new_frame, None, *params)
    flow = cv2.resize(flow * 2.0, (720, 480))
    return flow

def save_mask_parameters(mask_dict, mask_dict_path):
    """Save mask parameters to a JSON file."""
    with open(mask_dict_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(mask_dict, ensure_ascii=False, indent=4))

def draw_rectangle(frame, top_left, bottom_right, color, thickness, line_type):
    """Draw a rectangle on a frame."""
    cv2.rectangle(frame, top_left, bottom_right, color, thickness, line_type)

def save_video_with_mask(frames, output_path, fps=8):
    """Save a video with mask applied to frames."""
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

def save_sketch_with_mask(input_frame_path, output_frame_path, top_left, bottom_right, color, thickness, line_type):
    """Save a sketch frame with a mask applied."""
    input_frame = cv2.imread(input_frame_path)
    draw_rectangle(input_frame, top_left, bottom_right, color, thickness, line_type)
    cv2.imwrite(output_frame_path, input_frame)

def prepare_inter_mask(config, sketch_frame_id, temp_frms, root_dir):
    """Prepare mask and video for 'inter' bounding box type."""
    box_begin = config["box_begin"]
    box_end = config["box_end"]
    box_height = config["box_height"]
    box_width = config["box_width"]

    mask_save_dict = {
        "type": "inter",
        "box_begin": box_begin,
        "box_end": box_end,
        "box_height": box_height,
        "box_width": box_width,
    }
    mask_dict_path = os.path.join(root_dir, 'sketch_mask.json')
    save_mask_parameters(mask_save_dict, mask_dict_path)

    point_color = (243, 152, 0)
    thickness = 8
    line_type = 4

    sketch_mask = torch.zeros([49, 1, 480, 720])
    frames = []

    for i in range(49):
        src = temp_frms[i]
        box_begin_h = box_begin[0] + int((box_end[0] - box_begin[0]) / 49 * i)
        box_begin_w = box_begin[1] + int((box_end[1] - box_begin[1]) / 49 * i)

        draw_rectangle(src, (box_begin_w, box_begin_h), (box_begin_w + box_width, box_begin_h + box_height), point_color, thickness, line_type)
        frames.append(src)

        sketch_mask[i, :, box_begin_h:box_begin_h + box_height, box_begin_w:box_begin_w + box_width] = 1

        # if i == sketch_frame_id:
        #     input_frame_path = os.path.join(root_dir, f'sketches/{sketch_frame_id}.jpg')
        #     output_frame_path = os.path.join(root_dir, f'video_editing{sketch_frame_id}_mask.png')
        #     save_sketch_with_mask(input_frame_path, output_frame_path, (box_begin_w, box_begin_h), (box_begin_w + box_width, box_begin_h + box_height), point_color, thickness, line_type)

    output_path = os.path.join(root_dir, "editing_mask.mp4")
    save_video_with_mask(frames, output_path)

    sketch_mask_path = config["sketch_mask_path"]
    torch.save(sketch_mask, sketch_mask_path)

def prepare_flow_mask(config, sketch_frame_id, frames, root_dir):
    """Prepare mask and video for 'flow' bounding box type."""
    box_begin = config["box_begin"]
    box_height = config["box_height"]
    box_width = config["box_width"]

    mask_save_dict = {
        "type": "flow",
        "box_begin": box_begin,
        "box_height": box_height,
        "box_width": box_width,
    }
    mask_dict_path = os.path.join(root_dir, 'sketch_mask.json')
    save_mask_parameters(mask_save_dict, mask_dict_path)

    point1 = box_begin
    point2 = [box_begin[0] + box_height, box_begin[1] + box_width]

    point_color = (243, 152, 0)
    thickness = 8
    line_type = 4

    sketch_mask = torch.zeros([49, 1, 480, 720])
    sketch_mask[0, :, int(point1[0]):int(point2[0]), int(point1[1]):int(point2[1])] = 1

    box_frames = []
    frame_drawing_box = frames[0].copy()
    draw_rectangle(frame_drawing_box, (int(point1[1]), int(point1[0])), (int(point2[1]), int(point2[0])), point_color, thickness, line_type)
    box_frames.append(frame_drawing_box)

    # if 0 == sketch_frame_id:
    #     input_frame_path = os.path.join(root_dir, f'sketches/{sketch_frame_id}.jpg')
    #     output_frame_path = os.path.join(root_dir, f'video_editing{sketch_frame_id}_mask.png')
    #     save_sketch_with_mask(input_frame_path, output_frame_path, (int(point1[1]), int(point1[0])), (int(point2[1]), int(point2[0])), point_color, thickness, line_type)

    method = cv2.calcOpticalFlowFarneback
    params = [0.5, 3, 15, 3, 5, 1.2, 0]
    frame_interval = 4

    for i in range(1, len(frames)):
        if i % frame_interval == 1:
            if (i - 1 + frame_interval) > len(frames):
                frame_interval = len(frames) % frame_interval

            old_frame = frames[i - 1]
            new_frame = frames[i - 1 + frame_interval]
            flow = dense_optical_flow(method, old_frame, new_frame, params, to_gray=True)

            h_change = np.mean(flow[int(point1[0]):int(point2[0]), int(point1[1]):int(point2[1]), 1]) if point1[0] != point2[0] else 0
            w_change = np.mean(flow[int(point1[0]):int(point2[0]), int(point1[1]):int(point2[1]), 0]) if point1[1] != point2[1] else 0

            h_change = 0 if np.isnan(h_change) else h_change
            w_change = 0 if np.isnan(w_change) else w_change

        inter_value = 1.0 / frame_interval
        point1[0] += h_change * inter_value
        point1[1] += w_change * inter_value
        point2[0] += h_change * inter_value
        point2[1] += w_change * inter_value

        point1[0] = np.clip(point1[0], 0, 480)
        point2[0] = np.clip(point2[0], 0, 480)
        point1[1] = np.clip(point1[1], 0, 720)
        point2[1] = np.clip(point2[1], 0, 720)

        sketch_mask[i, :, int(point1[0]):int(point2[0]), int(point1[1]):int(point2[1])] = 1

        frame_drawing_box = frames[i].copy()
        draw_rectangle(frame_drawing_box, (int(point1[1]), int(point1[0])), (int(point2[1]), int(point2[0])), point_color, thickness, line_type)
        box_frames.append(frame_drawing_box)

        # if i == sketch_frame_id:
        #     input_frame_path = os.path.join(root_dir, f'sketches/{sketch_frame_id}.jpg')
        #     output_frame_path = os.path.join(root_dir, f'video_editing{sketch_frame_id}_mask.png')
        #     save_sketch_with_mask(input_frame_path, output_frame_path, (int(point1[1]), int(point1[0])), (int(point2[1]), int(point2[0])), point_color, thickness, line_type)

    output_path = os.path.join(root_dir, "editing_mask.mp4")
    save_video_with_mask(box_frames, output_path)

    sketch_mask_path = config["sketch_mask_path"]
    torch.save(sketch_mask, sketch_mask_path)

def main(config_path):
    """Main function to process the mask generation."""
    config = load_config(config_path)

    root_dir = config["root_dir"]
    video_path = config["video_path"]
    bounding_box_type = config["bounding_box_type"]
    sketch_frame_id = 48

    vr = VideoReader(uri=video_path, height=-1, width=-1)
    temp_frms = vr.get_batch(np.arange(0, len(vr))).asnumpy()

    if bounding_box_type == 'inter':
        prepare_inter_mask(config, sketch_frame_id, temp_frms, root_dir)
    else:
        prepare_flow_mask(config, sketch_frame_id, temp_frms, root_dir)

def parse_args(input_args=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Inference script for CogVideo with ControlNet.")
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file containing the parameters.",
    )
    return parser.parse_args(input_args) if input_args else parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)

