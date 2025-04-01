root_dir = "./editing_exp/single_man"

# Mask generation settings
bounding_box_type = "flow"
box_begin = [10, 200]
box_height = 200
box_width = 340

# Input paths
video_path = f"{root_dir}/man_editing.mp4"
sketch_paths = [
    f"{root_dir}/editing_1.png",
]
sketch_mask_path = f"{root_dir}/sketch_mask.pkl"

# Output directory
save_dir = f"{root_dir}/output"

# Prompt and control settings
validation_prompt = (
    "The video shows a man with a beard standing next to a black truck. He is wearing a black T-shirt. In the first frame, he is pointing at the truck. In the second frame, he is opening the door of the truck. In the third frame, he is sitting inside the truck. The truck is parked in a lot with trees in the background. The man appears to be in the process of getting into the truck. The style of the video is casual and informal."
)
control_frame_index = [[0],[0]]

# Model paths
controlnet_path = "Okrin/SketchVideo/sketchedit"
vae_path = "THUDM/CogVideoX-2b"
pipeline_path = "THUDM/CogVideoX-2b"

# Inference settings
guidance_scale = 25.0
num_inference_steps = 50
num_frames = 49
fps = 8

# Only support tow cases: 
# 1. Generate single example: seed=int, num_seeds=None
# 2. Generate multiple examples: seed=None, num_seeds=int
# seed = None
# num_seeds = 5
# inversion_fusion = False

seed = 5400
num_seeds = None
inversion_fusion = True
