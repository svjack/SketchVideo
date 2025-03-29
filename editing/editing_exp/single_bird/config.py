root_dir = "./editing_exp/single_bird"

# Mask generation settings
bounding_box_type = "flow"
box_begin = [100, 260]
box_height = 320
box_width = 300

# Input paths
video_path = f"{root_dir}/branches_editing.mp4"
sketch_paths = [
    f"{root_dir}/editing_3.png",
]
sketch_mask_path = f"{root_dir}/sketch_mask.pkl"

# Output directory
save_dir = f"{root_dir}/output"

# Prompt and control settings
validation_prompt = (
    "A vibrant magpie, its feathers a striking contrast of soft white and cool black, perches gracefully atop a slender branch of an ancient flower tree. The bird's delicate talons grip the bark with precision, as soft morning light filters through the leafless canopy, casting gentle shadows that dance around it. The magpie's head tilts slightly, its keen eye surveying the tranquil winter scene, while the rest of the forest lies quiet and still in the crisp, cool air of an early dawn."
)
control_frame_index = [[0],[0]]

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
# seed = None
# num_seeds = 5
# inversion_fusion = False

seed = 7231
num_seeds = None
inversion_fusion = True
