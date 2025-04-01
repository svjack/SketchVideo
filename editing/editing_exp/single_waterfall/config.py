root_dir = "./editing_exp/single_waterfall"

# Mask generation settings
bounding_box_type = "flow"
box_begin = [100, 250] # [h, w]
box_height = 350
box_width = 300

# Input paths
video_path = f"{root_dir}/waterfall_editing.mp4"
sketch_paths = [
    f"{root_dir}/editing_0.png",
]
sketch_mask_path = f"{root_dir}/sketch_mask.pkl"

# Output directory
save_dir = f"{root_dir}/output"

# Prompt and control settings
validation_prompt = (
    "A magnificent waterfall cascades down from a series of rocky cliffs. The waterfall occupies most of the area in the picture, which is very magnificent and spectacular. The water flow of the waterfall is very turbulent, with white splashes spanning the entire scene. The waterfall cascades from the top of the cliff all the way to the bottom of the lake."
)
control_frame_index = [[0],[0]]

# Model paths
controlnet_path = "Okrin/SketchVideo/sketchedit"
vae_path = "THUDM/CogVideoX-2b"
pipeline_path = "THUDM/CogVideoX-2b"

# Inference settings
guidance_scale = 6.0
num_inference_steps = 50
num_frames = 49
fps = 8

# Only support tow cases: 
# 1. Generate single example: seed=int, num_seeds=None
# 2. Generate multiple examples: seed=None, num_seeds=int

seed = 4535
num_seeds = None
inversion_fusion = True
