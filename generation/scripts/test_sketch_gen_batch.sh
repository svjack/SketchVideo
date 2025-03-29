# Batch example generation
python inference_ctrl_cogvideo_batch.py \
--input_dir "./results/ex13/test_input" \
--output_dir "./results/ex13/output_video_0_12" \
--controlnet_name "full" \
--control_frame_index "0,12" \
--control_checkpoint_path "Okrin/SketchVideo/sketchgen" \
--cogvideo_checkpoint_path "THUDM/CogVideoX-2b" \
--num_example 5 \