# Using Diffusion Priors for Video Amodal Segmentation

**CVPR 2025**

Official implementation of <strong>Using Diffusion Priors for Video Amodal Segmentation</strong>

[*Kaihua Chen*](https://www.linkedin.com/in/kaihuac/), [*Deva Ramanan*](https://www.cs.cmu.edu/~deva/), [*Tarasha Khurana*](https://www.cs.cmu.edu/~tkhurana/)

![diffusion-vas](assets/diffusion-vas.gif)

[**Paper**](https://arxiv.org/abs/2412.04623) | [**Project Page**](https://diffusion-vas.github.io)

## TODO 🤓

- [x] Release the checkpoint and inference code 
- [x] Release evaluation code for SAIL-VOS and TAO-Amodal
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

For **Depth Anything V2**'s checkpoints, download the Pre-trained Models (e.g., Depth-Anything-V2-Large) from **[this link](https://github.com/DepthAnything/Depth-Anything-V2)** and place them inside the `checkpoints/` folder.

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

Start with a video, use the **SAM2**'s [web demo](https://sam2.metademolab.com/) or its [codebase](https://github.com/facebookresearch/sam2) to segment the target object, and extract frames preferably at 8 FPS. Ensure that the output follows the same directory structure as examples from `demo_data/` before running inference.

## Evaluation

We currently support evaluation on **SAIL-VOS-2D** and **TAO-Amodal**.

### Download Datasets

- For **SAIL-VOS-2D**, follow the official instructions: https://sailvos.web.illinois.edu/_site/index.html  
- For **TAO-Amodal**, follow the instructions: https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal  

Additionally, download our curated annotations and precomputed evaluation results:

```bash
git clone https://huggingface.co/datasets/kaihuac/diffusion_vas_datasets
```

This includes:
- `diffusion_vas_sailvos_val.json`
- `diffusion_vas_tao_amodal_val.json`
- `tao_amodal_track_ids_abs2rel_val.json`
- Precomputed `eval_outputs/` folder

### Generate Evaluation Results

To evaluate the model, first generate result files using the scripts below. Alternatively, you can skip this step and directly use our precomputed results in `eval_outputs/`.

*Note: Please replace the paths in the commands with your own dataset and annotation paths.*

**SAIL-VOS-2D**
```bash
cd eval
python eval_diffusion_vas_sailvos.py \
    --eval_data_path /path/to/SAILVOS_2D/ \
    --eval_annot_path /path/to/diffusion_vas_sailvos_val.json \
    --eval_output_path /path/to/eval_outputs/
```

**TAO-Amodal**
```bash
python eval_diffusion_vas_tao_amodal.py \
    --eval_data_path /path/to/TAO/frames/ \
    --eval_annot_path /path/to/diffusion_vas_tao_amodal_val.json \
    --track_ids_path /path/to/tao_amodal_track_ids_abs2rel_val.json \
    --eval_output_path /path/to/eval_outputs/
```

### Compute Metrics

Once the result files are ready, run the metric scripts:

**SAIL-VOS-2D**
```bash
python metric_diffusion_vas_sailvos.py \
    --eval_data_path /path/to/SAILVOS_2D/ \
    --eval_annot_path /path/to/diffusion_vas_sailvos_val.json \
    --pred_annot_path /path/to/eval_outputs/diffusion_vas_sailvos_eval_results.json
```

**TAO-Amodal**
```bash
python metric_diffusion_vas_tao_amodal.py \
    --eval_data_path /path/to/TAO/frames/ \
    --eval_annot_path /path/to/diffusion_vas_tao_amodal_val.json \
    --track_ids_path /path/to/tao_amodal_track_ids_abs2rel_val.json \
    --pred_annot_path /path/to/eval_outputs/diffusion_vas_tao_amodal_eval_results.json

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



