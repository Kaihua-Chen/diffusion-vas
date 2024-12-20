# Using Diffusion Priors for Video Amodal Segmentation

[**Paper**](https://arxiv.org/abs/2412.04623) | [**Project Page**](https://diffusion-vas.github.io) | [**Video**](https://www.youtube.com/watch?v=ilHCPNaH7tY)

Using Diffusion Priors for Video Amodal Segmentation <br>
Kaihua Chen, Deva Ramanan, Tarasha Khurana <br>
Carnegie Mellon University

Abstract: Object permanence in humans is a fundamental cue that helps in understanding persistence of objects, even when they are fully occluded in the scene. Present day methods in object segmentation do not account for this _amodal_ nature of the world, and only work for segmentation of visible or _modal_ objects. Few amodal methods exist; single-image segmentation methods cannot handle high-levels of occlusions which are better inferred using temporal information, and multi-frame methods have focused solely on segmenting rigid objects. To this end, we propose to tackle video amodal segmentation by formulating it as a conditional generation task, capitalizing on the foundational knowledge in video generative models. Our method is simple; we repurpose these models to condition on a sequence of modal mask frames of an object along with contextual pseudo-depth maps, to learn which object boundary may be occluded and therefore, extended to hallucinate the complete extent of an object. This is followed by a content completion stage which is able to inpaint the occluded regions of an object.
We benchmark our approach alongside a wide array of state-of-the-art methods on four datasets and show a dramatic improvement of upto 13% for amodal segmentation in an object's occluded region.

<p align="center">
    <img src="assets/teaser.png">
</p>

## Updates

- Our [**paper**](https://arxiv.org/abs/2412.04623) is now available on arXiv!
- Code release coming soon... stay tuned! 🤓
- Check out our [**project page**](https://diffusion-vas.github.io) for more details!

## Citing us

```bibtex
@article{chen2024diffvas,
  title={Using Diffusion Priors for Video Amodal Segmentation},
  author={Kaihua Chen and Deva Ramanan and Tarasha Khurana},
  year={2024},
  archivePrefix={arXiv},
  eprint={2412.04623},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2412.04623}
}
```
