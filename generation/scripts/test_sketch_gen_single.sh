# Single example generation
python inference_ctrl_cogvideo.py \
--text_path "./results/ex1/test_input/car.txt" \
--image_paths "./results/ex1/test_input/car.png" \
--output_dir "./results/ex1/test_output" \
--controlnet_name "full" \
--control_frame_index "6" \
--control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
--cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
--seed 5990 \

# python inference_ctrl_cogvideo.py \
# --text_path "./results/ex1/test_input/cat2.txt" \
# --image_paths "./results/ex1/test_input/cat2.png" \
# --output_dir "./results/ex1/test_output" \
# --controlnet_name "full" \
# --control_frame_index "12" \
# --control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
# --cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
# --seed 791 \

# python inference_ctrl_cogvideo.py \
# --text_path "./results/ex1/test_input/girl3.txt" \
# --image_paths "./results/ex1/test_input/girl3.png" \
# --output_dir "./results/ex1/test_output" \
# --controlnet_name "full" \
# --control_frame_index "3" \
# --control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
# --cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
# --seed 8349 \

# python inference_ctrl_cogvideo.py \
# --text_path "./results/ex1/test_input/landscape.txt" \
# --image_paths "./results/ex1/test_input/landscape.png" \
# --output_dir "./results/ex1/test_output" \
# --controlnet_name "full" \
# --control_frame_index "9" \
# --control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
# --cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
# --seed 7418 \

# python inference_ctrl_cogvideo.py \
# --text_path "./results/ex1/test_input/ship.txt" \
# --image_paths "./results/ex1/test_input/ship.png" \
# --output_dir "./results/ex1/test_output" \
# --controlnet_name "full" \
# --control_frame_index "12" \
# --control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
# --cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
# --seed 9346 \

# python inference_ctrl_cogvideo.py \
# --text_path "./results/ex1/test_input/star_sky1.txt" \
# --image_paths "./results/ex1/test_input/star_sky1.png" \
# --output_dir "./results/ex1/test_output" \
# --controlnet_name "full" \
# --control_frame_index "0" \
# --control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
# --cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
# --seed 1246 \
