


<p align="center">
  <h1 align="center"> <img src="assets/cat.png" alt="Cambrian" width="23" height="auto"> GARF: Learning Generalizable 3D Reassembly </br> for Real-World Fractures </h1>
  <p align="center">
  A generalizable flow matching-based 3D reassembly method trained on 1.9 Million fractures, enabling precise real-world fragment pose alignment. 😊across extensive benchmarks, concise code with efficient performance.
  </p>
  <p align="center">
    <a href="https://jytime.github.io/data/VGGT_CVPR25.pdf" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
    </a>
    <a href="https://arxiv.org/abs/2503.11651"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
    <a href="https://vgg-t.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
    <a href='https://huggingface.co/spaces/facebook/vggt'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
  </p>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=90IoeJsAAAAJ">Sihang Li*</a>
    ·
    <a href="https://github.com/JDScript">Zeyu Jiang*</a>
    ·
    <a href="https://www.linkedin.com/in/grace-chen-37a974293/">Grace Chen†</a>
    ·
    <a href="https://uk.linkedin.com/in/chenyang-xu-755125181">Chenyang Xu†</a>
    ·
    <a href="https://github.com/kevintsq">Siqi Tan</a>
    ·
    <a href="https://github.com/kevintsq">Xue Wang</a>
    ·
    <a href="https://irvingf7.github.io/">Irving Fang</a>
    ·
    <a href="https://scholar.google.com/citations?user=aEmILscAAAAJ&hl=en">Kristof Zyskowski</a>
    ·
    <a href="https://scholar.google.com/citations?user=lo1VSPUAAAAJ&hl=en">Shannon McPherron</a>
    ·
    <a href="https://scholar.google.com/citations?user=JqLHsvYAAAAJ&hl=en">Radu Iovita</a>
    ·
    <a href="https://scholar.google.com/citations?hl=en&user=YeG8ZM0AAAAJ">Chen Feng✉</a>
    ·
    <a href="https://jingz6676.github.io/">Jing Zhang✉</a>
  </p>
  <p align="center">
    *, † Equal contribution ✉ Corresponding author

<p align="center">
    <img src="assets/main.png" alt="Main Figure" width="100%" />
</p>

  <div align="center"></div>

</p>


## News 🚀🚀🚀
- `2025/03/25`: We release the [GARF](https://huggingface.co/collections/IPEC-COMMUNITY/foundation-vision-language-action-model-6795eb96a9c661f90236acbb), which achieves state-of-the-art performance across a diverse range of synthetic and real-world benchmarks. Try our [demo](https://garf-demo.pages.dev/) on your own data! 

## 📖 Table of Contents

- [📄 Document](#Document)
- [🤗 Model Zoo](#data-preparation)
- [✅ Evaluation Performance](#Performance)
- [🙋 FAQs](#faq)
- [Citation](#citation)
- [License](#license)

## 📄 Document


### ⏩ **Installation**

### 💾 **Data Preparation**

### 🎯 **Evaluation**

### ⭐ **Stage 1: Fracture-aware Pretraining**

### ⭐ **Stage 2: Flow-matching Training**

### ⭐ **(Optional) Stage 3: LoRA-based Fine-tuning**


<!-- ### 🎯💾⏩🌈🌟⭐🥑♣️♠️♟️🎮✨🏷️📍📌✈️ Data Preparation -->

## 😺 Model Zoo


<table>
  <tr>
    <th>Model Name</th>
    <th>Backbone</th>
    <th>Model</th>
    <th>Note</th>
  </tr>
  <tr>
    <td>GARF-mini</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">spatialvla-4b-224-pt</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal">spatialvla-4b-224-sft-fractal</a></td>
    <td>fine-tuning on the fractal dataset, testing on simple-env google-robot, TABLE II ine-tuning</a></td>
  </tr>
  <tr>
    <td>GARF-mini-diffusion</td>
    <td><a href="https://huggingface.co/google/paligemma2-3b-pt-224">google/paligemma2-3b-pt-224</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">spatialvla-4b-224-pt</a></td>
    <td>pretrained on openx and rh20t, TABLE I and II zero-shot, Fig.5 and 7</td>
  </tr>
  <tr>
    <td>GARF</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">spatialvla-4b-224-pt</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-mix-224-pt">spatialvla-4b-mix-224-pt</a></td>
    <td>fine-tuning on the fractal and bridge mixture dataset, Fig.5 and 7</a></td>
  </tr>
  <tr>
    <td>GARF-Pro</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">spatialvla-4b-224-pt</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge">spatialvla-4b-224-sft-bridge</a></td>
    <td>fine-tuning on the bridge dataset, testing on simple-env widowx-robot, TABLE I fine-tuning</a></td>
  </tr>
</table>






  <!-- ```bibtex
    @inproceedings{li2025garf,
    title={VGGT: Visual Geometry Grounded Transformer},
    author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
    }
``` -->