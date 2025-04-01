# Single example generation
python inference_ctrl_cogvideo.py \
--text_path "./results/ex2/test_input/dog3.txt" \
--image_paths "./results/ex2/test_input/dog3_1.png,./results/ex2/test_input/dog3_2.png" \
--output_dir "./results/ex2/test_output" \
--controlnet_name "full" \
--control_frame_index "0,12" \
--control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
--cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
--seed 7191 \

# python inference_ctrl_cogvideo.py \
# --text_path "./results/ex2/test_input/cake2.txt" \
# --image_paths "./results/ex2/test_input/cake2_1.png,./results/ex2/test_input/cake2_2.png" \
# --output_dir "./results/ex2/test_output" \
# --controlnet_name "full" \
# --control_frame_index "0,12" \
# --control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
# --cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
# --seed 292 \

# python inference_ctrl_cogvideo.py \
# --text_path "./results/ex2/test_input/castle.txt" \
# --image_paths "./results/ex2/test_input/castle_1.png,./results/ex2/test_input/castle_2.png" \
# --output_dir "./results/ex2/test_output" \
# --controlnet_name "full" \
# --control_frame_index "3,9" \
# --control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
# --cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
# --seed 1770 \

# python inference_ctrl_cogvideo.py \
# --text_path "./results/ex2/test_input/cat.txt" \
# --image_paths "./results/ex2/test_input/cat_1.png,./results/ex2/test_input/cat_2.png" \
# --output_dir "./results/ex2/test_output" \
# --controlnet_name "full" \
# --control_frame_index "6,12" \
# --control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
# --cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
# --seed 3776 \
