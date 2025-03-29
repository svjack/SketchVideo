root_dir = "./editing_exp/two_girl"

# Mask generation settings
bounding_box_type = "flow"
box_begin = [10, 150] # [h, w]
box_height = 220
box_width = 450

# Input paths
video_path = f"{root_dir}/girl_editing.mp4"
sketch_paths = [
    f"{root_dir}/editing_0.png",
    f"{root_dir}/editing_12.png",
]
sketch_mask_path = f"{root_dir}/sketch_mask.pkl"

# Output directory
save_dir = f"{root_dir}/output"

# Prompt and control settings
validation_prompt = (
    "A close-up shot captures the innocent, yet adventurous expression of a young, blonde girl, her eyes a shimmering shade of blue. She's adorned in a classic khaki canvas top hat, casting a gentle shadow over her bright, curious eyes. The scene is one of quiet wonder, with the girl's face conveying a mix of mischief and wonder, as if she's about to embark on a grand, unknown journey."
)
control_frame_index = [[0,3],[0,3]]

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

seed = 19
num_seeds = None
inversion_fusion = True
