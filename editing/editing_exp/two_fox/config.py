root_dir = "./editing_exp/two_fox"

# Mask generation settings
bounding_box_type = "flow"
box_begin = [100, 230] # [h, w]
box_height = 300
box_width = 270

# Input paths
video_path = f"{root_dir}/grass_editing.mp4"
sketch_paths = [
    f"{root_dir}/editing_12.png",
    f"{root_dir}/editing_36.png",
]
sketch_mask_path = f"{root_dir}/sketch_mask.pkl"

# Output directory
save_dir = f"{root_dir}/output"

# Prompt and control settings
validation_prompt = (
    "A small, light white fox with a fluffy tail and alert ears sits gracefully atop a lush green meadow, its head initially facing the viewer with a curious gaze. Gradually, the little fox tilts its head to the left, its eyes glinting with a mix of curiosity and wariness, as it surveys the surroundings. The soft sunlight filters through the nearby trees, casting dappled shadows that dance across the grass and the fox's fur, creating a tranquil and picturesque scene. The grassland in the background presents a bright yellow green color."
)
control_frame_index = [[3,9],[3,9]]

# Model paths
controlnet_path = "Okrin/SketchVideo/sketchedit"
vae_path = "THUDM/CogVideoX-2b"
pipeline_path = "THUDM/CogVideoX-2b"

# Inference settings
guidance_scale = 15.0
num_inference_steps = 50
num_frames = 49
fps = 8

# Only support tow cases: 
# 1. Generate single example: seed=int, num_seeds=None
# 2. Generate multiple examples: seed=None, num_seeds=int
# seed = None
# num_seeds = 5
# inversion_fusion = False

seed = 3867
num_seeds = None
inversion_fusion = True
