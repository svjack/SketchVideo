import decord
from decord import VideoReader
import numpy as np
import cv2
import random
import os
import tqdm
import sys
from PIL import Image
import imageio

sys.path.append('../')
from annotator.lineart import LineartDetector

def pad_last_frame(frames, target_num_frames):
    """
    Pads the last frame of the video to match the target number of frames.
    """
    if frames.shape[0] < target_num_frames:
        last_frame = frames[-1:]
        padding = np.repeat(last_frame, target_num_frames - frames.shape[0], axis=0)
        padded_frames = np.concatenate([frames, padding], axis=0)
        return padded_frames
    else:
        return frames[:target_num_frames]

def resize_for_rectangle_crop(frames, target_size, crop_mode="random"):
    """
    Resizes frames to fit the target rectangle size and crops them.
    """
    frame_height, frame_width = frames.shape[1:3]
    target_height, target_width = target_size

    # Determine resizing dimensions
    if frame_width / frame_height > target_width / target_height:
        new_width = int(frame_width * target_height / frame_height)
        resized_frames = np.array([cv2.resize(frame, (new_width, target_height)) for frame in frames])
    else:
        new_height = int(frame_height * target_width / frame_width)
        resized_frames = np.array([cv2.resize(frame, (target_width, new_height)) for frame in frames])

    # Calculate cropping offsets
    crop_height, crop_width = resized_frames.shape[1:3]
    delta_h, delta_w = crop_height - target_height, crop_width - target_width

    if crop_mode == "random":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif crop_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise ValueError(f"Unsupported crop_mode: {crop_mode}")

    # Crop frames
    cropped_frames = resized_frames[:, top:top + target_height, left:left + target_width]
    return cropped_frames

def nearest_smaller_4k_plus_1(n):
    """
    Finds the nearest smaller number that satisfies 4k+1.
    """
    remainder = n % 4
    return n - remainder + 1 if remainder != 0 else n - 3

def resize_video(video_path, fps=8, max_frames=49, skip_frames=0):
    """
    Resizes a video to a fixed number of frames and dimensions.
    """
    vr = VideoReader(uri=video_path)
    actual_fps = vr.get_avg_fps()
    total_frames = len(vr)

    # Determine frame indices
    if (total_frames - skip_frames) / actual_fps * fps > max_frames:
        num_frames = max_frames
        start = int(skip_frames)
        end = int(start + num_frames / fps * actual_fps)
        indices = np.linspace(start, end, num_frames, endpoint=False).astype(int)
    else:
        start = int(skip_frames)
        end = int(total_frames - skip_frames)
        num_frames = nearest_smaller_4k_plus_1(end - start)
        indices = np.linspace(start, start + num_frames, num_frames, endpoint=False).astype(int)

    # Extract and process frames
    frames = vr.get_batch(indices).asnumpy()
    frames = pad_last_frame(frames, num_frames)
    frames = resize_for_rectangle_crop(frames, target_size=[480, 720], crop_mode="center")
    return frames

def save_sketch_frames(frames, output_dir, detector, step=4):
    """
    Saves sketch frames generated from the input frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(0, len(frames), step):
        resized_frame = cv2.resize(frames[i], (360, 240))
        sketch_map = detector(resized_frame, coarse=False)
        sketch_map = cv2.resize(sketch_map, (720, 480))
        sketch_map[sketch_map < 200] = 0
        sketch_map[sketch_map > 200] = 255

        sketch_path = os.path.join(output_dir, f"{i}.jpg")
        Image.fromarray(sketch_map).save(sketch_path)

def save_video(frames, output_path, fps=8):
    """
    Saves frames as a video.
    """
    writer = imageio.get_writer(output_path, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()

if __name__ == "__main__":
    input_path = './editing_exp/bird/wild_video.mp4'
    output_path = './editing_exp/bird/original_editing.mp4'
    frames_dir = './editing_exp/bird/sketches/'

    # Resize video and process frames
    frames = resize_video(input_path)
    frames = frames.astype(np.uint8)

    # Detect sketches
    device = 'cpu'
    lineart_detector = LineartDetector(device)
    save_sketch_frames(frames, frames_dir, lineart_detector)

    # Save the final video
    save_video(frames, output_path)

