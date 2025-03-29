root_dir = "./editing_exp/two_temple"

# Mask generation settings
bounding_box_type = "inter"
box_begin = [10, 200]
box_end = [5, 180]
box_height = 200
box_width = 550

# Input paths
video_path = f"{root_dir}/temple_editing.mp4"
sketch_paths = [
    f"{root_dir}/editing_0.png",
    f"{root_dir}/editing_48.png",
]
sketch_mask_path = f"{root_dir}/sketch_mask.pkl"

# Output directory
save_dir = f"{root_dir}/output"

# Prompt and control settings
validation_prompt = (
    "An ancient beige white temple, crafted from marble with a subtle yellow patina, stands majestically amidst a serene landscape. It boasts an array of towering pillars, each meticulously carved, supporting a grand, beige white roof. The temple's weathered surface reveals the whispers of time, marked by intricate cracks and the subtle discolorations that speak to its storied past."
)
control_frame_index = [[0,12],[0,12]]

# Model paths
controlnet_path = "Okrin/SketchVideo/sketchedit"
vae_path = "THUDM/CogVideoX-2b"
pipeline_path = "THUDM/CogVideoX-2b"

# Inference settings
guidance_scale = 10.0
num_inference_steps = 50
num_frames = 49
fps = 8

# Only support tow cases: 
# 1. Generate single example: seed=int, num_seeds=None
# 2. Generate multiple examples: seed=None, num_seeds=int
# seed = None
# num_seeds = 5
# inversion_fusion = False

seed = 2180
num_seeds = None
inversion_fusion = True
