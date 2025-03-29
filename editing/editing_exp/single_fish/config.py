root_dir = "./editing_exp/single_fish"

# Mask generation settings
bounding_box_type = "inter"
box_begin = [100, 100]
box_end = [150, 400]
box_height = 200
box_width = 250

# Input paths
video_path = f"{root_dir}/ocean_input_editing.mp4"
sketch_paths = [
    f"{root_dir}/editing_24.png",
]
sketch_mask_path = f"{root_dir}/sketch_mask.pkl"

# Output directory
save_dir = f"{root_dir}/output"

# Prompt and control settings
validation_prompt = (
    "The ornamental fish swims from left to right. A mesmerizing ornamental fish glides through the inky depths of the ocean, its vibrant scales shimmering with hues of electric blue, fiery orange, and iridescent green. With a sleek, flattened body that navigates the vast underwater expanse. Its delicate fins, adorned with intricate patterns, flutter and sway like rhythmic poetry in motion, leaving a trail of iridescence in the silent sea."
)
control_frame_index = [[6],[6]]

# Model paths
controlnet_path = "Okrin/SketchVideo/sketchedit"
vae_path = "THUDM/CogVideoX-2b"
pipeline_path = "THUDM/CogVideoX-2b"

# Inference settings
guidance_scale = 20.0
num_inference_steps = 50
num_frames = 49
fps = 8

# Only support tow cases: 
# 1. Generate single example: seed=int, num_seeds=None
# 2. Generate multiple examples: seed=None, num_seeds=int

seed = 7622
num_seeds = None
inversion_fusion = True
