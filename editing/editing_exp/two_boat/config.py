root_dir = "/mnt/new/liufenglin/code/SketchVideo_ori/editing/results/editing_exp/two_boat"

# Input paths
video_path = f"{root_dir}/ocean_editing.mp4"
sketch_paths = [
    f"{root_dir}/editing_0.png",
    f"{root_dir}/editing_48.png",
]
sketch_mask_path = f"{root_dir}/sketch_mask.pkl"

# Output directory
save_dir = f"{root_dir}/output"

# Prompt and control settings
validation_prompt = (
    "From a bird's-eye view, a sleek sailboat glides gracefully across a calm azure sea. It travels from the left horizon to the right. The coastline in the distance is a mere outline of lush greenery against the vast expanse of the tranquil ocean."
)
control_frame_index = [[0,12],[0,12]]

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
seed = 338
num_seeds = None

inversion_fusion = True
