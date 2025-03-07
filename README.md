# Using Diffusion Priors for Video Amodal Segmentation

**CVPR 2025**

Official implementation of <strong>Using Diffusion Priors for Video Amodal Segmentation</strong>

[*Kaihua Chen*](https://www.linkedin.com/in/kaihuac/), [*Deva Ramanan*](https://www.cs.cmu.edu/~deva/), [*Tarasha Khurana*](https://www.cs.cmu.edu/~tkhurana/)

![diffusion-vas](assets/diffusion-vas.gif)

[**Paper**](https://arxiv.org/abs/2412.04623) | [**Project Page**](https://diffusion-vas.github.io)

## TODO 🤓

- [x] Release the checkpoint and inference code 
- [ ] Release evaluation code for SAIL-VOS and TAO-Amodal
- [ ]  Release fine-tuning code for Diffusion-VAS

## Getting Started

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/Kaihua-Chen/diffusion-vas
cd diffusion-vas
```

#### 2. Create and activate a virtual environment

```bash
conda create --name diffusion_vas python=3.10
conda activate diffusion_vas
pip install -r requirements.txt
```

### Download Checkpoints

We provide our Diffusion-VAS checkpoints finetuned on SAIL-VOS on Hugging Face. To download them, run:

```bash
mkdir checkpoints
cd checkpoints
git lfs install
git clone https://huggingface.co/kaihuac/diffusion-vas-amodal-segmentation
git clone https://huggingface.co/kaihuac/diffusion-vas-content-completion
cd ..
```
*Note: Ignore any Windows-related warnings when downloading.*

For **Depth Anything V2**'s checkpoints, download the Pre-trained Models (e.g., ViT-L) from **[this link](https://github.com/DepthAnything/Depth-Anything-V2)** and place them inside the `checkpoints/` folder.

### Inference

To run inference, simply execute:

```bash
python demo.py
```

This will infer the birdcage example from `demo_data/`.

To try different examples, modify the `seq_name` argument:

```bash
python demo.py --seq_name <your_sequence_name>
```

You can also change the checkpoint path, data output paths, and other parameters as needed.

### Using custom data

Start with a video, use the **SAM2**'s [web demo](https://sam2.metademolab.com/) or its [codebase](https://github.com/facebookresearch/sam2) to segment the target object, and extract frames preferably at 8 FPS. Ensure that the output follows the same directory structure as demo_data/ before running inference.

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@inproceedings{chen2025diffvas,
      title={Using Diffusion Priors for Video Amodal Segmentation},
      author={Kaihua Chen and Deva Ramanan and Tarasha Khurana},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2025}
}
```



