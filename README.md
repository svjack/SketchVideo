```bash
sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg

conda create -n sketchvideo python=3.10
conda activate sketchvideo
pip install ipykernel  # 安装ipykernel
python -m ipykernel install --user --name sketchvideo --display-name "sketchvideo"  # 注册Jupyter kernel

git clone https://github.com/svjack/SketchVideo && cd SketchVideo
pip install -r requirements.txt

git clone https://huggingface.co/THUDM/CogVideoX-2b
git clone https://huggingface.co/Okrin/SketchVideo
```

## ***SketchVideo: Sketch-based Video Generation and Editing***
<!-- ![](./assets/logo_long.png#gh-light-mode-only){: width="50%"} -->
<!-- ![](./assets/logo_long_dark.png#gh-dark-mode-only=100x20) -->
<div align="center">
<img src='assets/logo.png' style="height:100px"></img>

<a href='https://arxiv.org/abs/2503.23284'><img src='https://img.shields.io/badge/arXiv-2405.17933-b31b1b.svg'></a> &nbsp;
<a href='http://geometrylearning.com/SketchVideo/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
<a href='https://www.youtube.com/watch?v=eo5DNiaGgiQ'><img src='https://img.shields.io/badge/Youtube-Video-b31b1b.svg'></a><br>

<strong> CVPR 2025</strong>


</div>
 
## &#x1F680; Introduction

We propose SketchVideo, which aim to achieve sketch-based spatial and motion control for video generation and support fine-grained editing of real or synthetic videos.  Please check our project page and paper for more information. <br>


### 1. Sketch-based Video Generation

#### 1.1 One keyframe Input Sketch
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input frame</td>
        <td>Generated video</td>
        <td>Input frame</td>
        <td>Generated video</td>
    </tr>

  <tr>
  <td>
    <img src=./generation/results/ex1/test_input/star_sky1.png width="200">
  </td>
  <td>
    <img src=assets/gen_one_frame/star_sky1.gif width="200">
  </td>
  <td>
    <img src=./generation/results/ex1/test_input/landscape.png width="200">
  </td>
   <td>
    <img src=assets/gen_one_frame/landscape.gif width="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=./generation/results/ex1/test_input/cat2.png width="200">
  </td>
  <td>
    <img src=assets/gen_one_frame/cat2.gif width="200">
  </td>
  <td>
    <img src=./generation/results/ex1/test_input/girl3.png width="200">
  </td>
   <td>
    <img src=assets/gen_one_frame/girl3.gif width="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=./generation/results/ex1/test_input/car.png width="200">
  </td>
  <td>
    <img src=assets/gen_one_frame/car.gif width="200">
  </td>
  <td>
    <img src=./generation/results/ex1/test_input/ship.png width="200">
  </td>
   <td>
    <img src=assets/gen_one_frame/ship.gif width="200">
  </td>
  </tr>


</table>

#### 1.2 Two keyframe Input Sketches
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input frame</td>
        <td>Generated video</td>
        <td>Input frame</td>
        <td>Generated video</td>
    </tr>

  <tr>
  <td>
    <img src=assets/gen_two_frame/cat_sketch.png width="200">
  </td>
  <td>
    <img src=assets/gen_two_frame/cat.gif width="200">
  </td>
  <td>
    <img src=assets/gen_two_frame/dog_sketch.png width="200">
  </td>
   <td>
    <img src=assets/gen_two_frame/dog.gif width="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/gen_two_frame/cake_sketch.png width="200">
  </td>
  <td>
    <img src=assets/gen_two_frame/cake2.gif width="200">
  </td>
  <td>
    <img src=assets/gen_two_frame/castle_sketch.png width="200">
  </td>
   <td>
    <img src=assets/gen_two_frame/castle.gif width="200">
  </td>
  </tr>



</table>

### 2. Sketch-based Video Editing
#### 2.1 One keyframe Input Sketch
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input sketch</td>
        <td>Original video</td>
        <td>Generated video</td>
    </tr>

  <tr>
  <td>
    <img src=assets/editing_one_frame/waterfall.png width="250">
  </td>
  <td>
    <img src=assets/editing_one_frame/waterfall_input.gif width="250">
  </td>
  <td>
    <img src=assets/editing_one_frame/waterfall_output.gif width="250">
  </td>
  </tr>


  <tr>
  <td>
    <img src=assets/editing_one_frame/man.png width="250">
  </td>
  <td>
    <img src=assets/editing_one_frame/man_input.gif width="250">
  </td>
  <td>
    <img src=assets/editing_one_frame/man_output.gif width="250">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/editing_one_frame/fish.png width="250">
  </td>
  <td>
    <img src=assets/editing_one_frame/fish_input.gif width="250">
  </td>
  <td>
    <img src=assets/editing_one_frame/fish_output.gif width="250">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/editing_one_frame/bird.png width="250">
  </td>
  <td>
    <img src=assets/editing_one_frame/bird_input.gif width="250">
  </td>
  <td>
    <img src=assets/editing_one_frame/bird_output.gif width="250">
  </td>
  </tr>

</table>


#### 2.2 Two keyframe Input Sketches
<table class="center">
    <tr style="font-weight: bolder;text-align:center;">
        <td>Input sketch 1</td>
        <td>Input sketch 2</td>
        <td>Original video</td>
        <td>Generated video</td>
    </tr>

  <tr>
  <td>
    <img src=assets/editing_two_frame/boat/editing_0.png width="200">
  </td>
  <td>
    <img src=assets/editing_two_frame/boat/editing_48.png width="200">
  </td>
  <td>
    <img src=assets/editing_two_frame/boat/ocean_input.gif width="200">
  </td>
   <td>
    <img src=assets/editing_two_frame/boat/ocean_output.gif width="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/editing_two_frame/temple/editing_0.png width="200">
  </td>
  <td>
    <img src=assets/editing_two_frame/temple/editing_48.png width="200">
  </td>
  <td>
    <img src=assets/editing_two_frame/temple/temple_input.gif width="200">
  </td>
   <td>
    <img src=assets/editing_two_frame/temple/temple_output.gif width="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/editing_two_frame/girl/editing_0.png width="200">
  </td>
  <td>
    <img src=assets/editing_two_frame/girl/editing_12.png width="200">
  </td>
  <td>
    <img src=assets/editing_two_frame/girl/girl_input.gif width="200">
  </td>
   <td>
    <img src=assets/editing_two_frame/girl/girl_output.gif width="200">
  </td>
  </tr>

  <tr>
  <td>
    <img src=assets/editing_two_frame/fox/editing_12.png width="200">
  </td>
  <td>
    <img src=assets/editing_two_frame/fox/editing_36.png width="200">
  </td>
  <td>
    <img src=assets/editing_two_frame/fox/fox_input.gif width="200">
  </td>
   <td>
    <img src=assets/editing_two_frame/fox/fox_output.gif width="200">
  </td>
  </tr>


</table>







## 📝 Changelog
- __[2025.04.01]__: 🔥🔥 Release code and model weights.
- __[2025.03.30]__: Launch the project page and update the arXiv preprint.
<br>


## 🧰 Models

|Model|Resolution|GPU Mem. & Inference Time (A100, ddim 50steps)|Checkpoint|
|:---------|:---------|:--------|:--------|
|SketchGen|720x480| ~21G & 95s |[Hugging Face](https://huggingface.co/Okrin/SketchVideo/tree/main/sketchgen)|
|SketchEdit|720x480| ~23G & 230s |[Hugging Face](https://huggingface.co/Okrin/SketchVideo/tree/main/sketchedit)|

Our method is built based on pretrained [CogVideo-2b](https://github.com/THUDM/CogVideo) model. We add an additional sketch conditional network for sketch-based generation and editing. 

Currently, our SketchVideo can support generating videos of up to 49 frames with a resolution of 720x480. For generation, we assume the sketches have a resolution of 720x480. For editing, we assume the input video has 49 frames with a resolution of 720x480.

The inference time can be reduced by using fewer DDIM steps.



## ⚙️ Setup

### Install Environment via Anaconda (Recommended)
```bash
conda create -n sketchvideo python=3.10
conda activate sketchvideo
pip install -r requirements.txt
```
Notably, `diffusers==0.30.1` is required. 

## 💫 Inference
### 1. Sketch-based Video Generation

Download pretrained SketchGen network [[hugging face](https://huggingface.co/Okrin/SketchVideo/tree/main/sketchgen)] and pretrained CogVideo-2b [[hugging face](https://huggingface.co/THUDM/CogVideoX-2b)] video generation model. Then, modify the `--control_checkpoint_path` and `--cogvideo_checkpoint_path` in scripts to corresponding paths. 

Generate video based on single keyframe sketch. 
```bash
cd generation
sh scripts/test_sketch_gen_single.sh
```

Generate video based on two keyframe sketches. 
```bash
cd generation
sh scripts/test_sketch_gen_two.sh
```

### 2. Sketch-based Video Editing

Download pretrained SketchEdit network [[hugging face](https://huggingface.co/Okrin/SketchVideo/tree/main/sketchedit)] and pretrained CogVideo-2b [[hugging face](https://huggingface.co/THUDM/CogVideoX-2b)] video generation model. Then, for each editing example, modify the `config.py` in `editing/editing_exp` folder. Change `controlnet_path` into SketchEdit weights path, and `vae_path, pipeline_path` into CogVideo weights path. 

Edit video based on keyframe sketches. 
```bash
cd editing
sh scripts/test_sketch_edit.sh
```

It contains the editing examples based on one or two keyframe sketches. 

## 😉 Citation
Please consider citing our paper if our code is useful:
```bib
@inproceedings{Liu2025sketchvideo,
  author    = {Liu, Feng-Lin and Fu, Hongbo and Wang, Xintao and Ye, Weicai and Wan, Pengfei and Zhang, Di and Gao, Lin},
  title     = {SketchVideo: Sketch-based Video Generation and Editing},
  booktitle    = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition},
  publisher    = {{IEEE}},
  year         = {2025},
}
```


## 🙏 Acknowledgements
We thanks the projects of video generation models [CogVideoX](https://github.com/THUDM/CogVideo) and [ControlNet](https://github.com/lllyasviel/ControlNet). Our code introduction is modified from [ToonCrafter](https://github.com/Doubiiu/ToonCrafter/tree/main) template.

<a name="disc"></a>
## 📢 Disclaimer
Our framework achieves interesting sketch-based video generation and editing, but due to the variaity of generative video prior, the success rate is not guaranteed. Different random seeds can be tried to generate the best video generation results. 

This project strives to impact the domain of AI-driven video generation positively. Users are granted the freedom to create videos using this tool, but they are expected to comply with local laws and utilize it responsibly. The developers do not assume any responsibility for potential misuse by users.
****
