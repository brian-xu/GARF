


<p align="center">
  <h1 align="center"> <img src="assets/cat.png" alt="Cambrian" width="23" height="auto"> GARF: Learning Generalizable 3D Reassembly </br> for Real-World Fractures </h1>
  <p align="center">
  A generalizable flow matching-based 3D reassembly method trained on 1.9 Million fractures, enabling precise real-world fragment pose alignment. 😊Achieves strong performance across extensive benchmarks, concise code with efficient performance.
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


## 🔊 News 
- `2025/03/25`: We release the [GARF](https://huggingface.co/collections/IPEC-COMMUNITY/foundation-vision-language-action-model-6795eb96a9c661f90236acbb), which achieves state-of-the-art performance across a diverse range of synthetic and real-world benchmarks. Try our [demo](https://garf-demo.pages.dev/) on your own data! 

## 📖 Table of Contents

- [📄 Document](#Document)
- [😺 Model Zoo](#data-preparation)
- [✅ Evaluation Performance](#Performance)
- [🙋 FAQs](#faq)
- [Citation](#citation)
- [License](#license)


## 📄 Document


### ⏩ **Installation**
We recommend using [uv](https://docs.astral.sh/uv/) to manage the dependencies. Follow the instructions [here](https://docs.astral.sh/uv/installation) to install uv. Then, simply run
```bash
uv sync
uv sync --extra post
source ./venv/bin/activate
```
to install the dependencies and activate the virtual environment. Please be noted that `flash-attn` requires CUDA 12.0 or above and you may fail to install it if you are using an older version of CUDA and NVCC.

### 💾 **Data Preparation**
We will soon provide the script to process the raw Breaking Bad dataset into our hdf5 format, right now, you can directly download our processed dataset from following links. Fractuna dataset will be released soon.
<table>
  <tr>
    <th>Dataset</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>Breaking Bad</td>
    <td><a href="https://jdscript-my.sharepoint.com/:f:/g/personal/shared_jdscript_app/EqEvBJxkWcJOpLDqLTaYiQgBayhtJWEzwO7ftRUf6dMBMw?e=oREaca" target="_blank">OneDrive</a></td>
  </tr>
  <tr>
    <td>Breaking Bad Volume Constrained</td>
    <td><a href="https://jdscript-my.sharepoint.com/:f:/g/personal/shared_jdscript_app/EqEvBJxkWcJOpLDqLTaYiQgBayhtJWEzwO7ftRUf6dMBMw?e=oREaca" target="_blank">OneDrive</a></td>
  </tr>
  <tr>
    <td>Breaking Bad Other</td>
    <td><a href="https://jdscript-my.sharepoint.com/:f:/g/personal/shared_jdscript_app/EqEvBJxkWcJOpLDqLTaYiQgBayhtJWEzwO7ftRUf6dMBMw?e=oREaca" target="_blank">OneDrive</a></td>
  </tr>
</table>

### 🎯 **Evaluation**
We provide the evaluation script in `scripts/eval.sh`.
### ⭐ **Stage 1: Fracture-aware Pretraining**
```bash
python train.py \
    experiment=pretraining_frac_seg \
    experiment_name=pretraining \
    data.categories="['everyday']" \
    project_name="GARF" \
    trainer.num_nodes=$NUM_NODES \
    data.data_root=./breaking_bad_vol.hdf5 \
    data.num_workers=8 \
    data.batch_size=32 \
    data.multi_ref=True \
    tags='["pretraining", 'everyday']' \
    ckpt_path=./xxx # to resume training
```
### ⭐ **Stage 2: Flow-matching Training**
```bash
python train.py \
    experiment=denoiser_flow_matching \
    experiment_name=denoiser \
    data.categories="['everyday']" \
    project_name="GARF" \
    trainer.num_nodes=$NUM_NODES \
    data.data_root=./breaking_bad_vol.hdf5 \
    data.num_workers=8 \
    data.batch_size=32 \
    data.multi_ref=True \
    tags='["denoiser", 'everyday']' \
    model.feature_extractor_ckpt=output/feature_extractor.ckpt \ # load the pretrained feature extractor
    ckpt_path=./xxx # to resume training
```
### ⭐ **(Optional) Stage 3: LoRA-based Fine-tuning**
```bash
python train.py \
    experiment=finetune \
    experiment_name=finetune \
    data.categories="['egg']" \
    project_name="GARF" \
    trainer.num_nodes=$NUM_NODES \
    data.data_root=./finetune_egg.hdf5 \
    data.num_workers=8 \
    data.batch_size=32 \
    data.multi_ref=True \
    tags='["finetune", 'egg']' \
    ckpt_path=./xxx \
    finetuning=true
```
### 📂 **Deploy Your Method**

### 🎮 **Visualization**

<!-- ### 🎮✒️📂🗂️📝📦🎯💾⏩🌈🌟⭐🥑♣️♠️♟️🎮✨🏷️📍📌✈️ Data Preparation -->

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
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">PTv3-E</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal">GARF-mini-E-FM</a></td>
    <td>pretrained on everyday subset of Breaking Bad with Flow-matching model. </a></td>
  </tr>
  <tr>
    <td>GARF-mini-diffusion</td>
    <td><a href="https://huggingface.co/google/paligemma2-3b-pt-224">PTv3-E</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">GARF-mini-E-Diff</a></td>
    <td>replace the Flow-matching model with Diffusion model</td>
  </tr>
  <tr>
    <td>GARF</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">PTv3-EAO</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-mix-224-pt">GARF-EAO-FM</a></td>
    <td>large-scale trained on everyday+artifact+other subsets of Breaking Bad for both backbone and Flow-matching (cost most time!)</a></td>
  </tr>
  <!-- <tr>
    <td>GARF-Pro</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">PTv3-E</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge">GARF-Pro-EAO-FM</a></td>
    <td>train Flow-matching model with everyday+artifact+other subsets</a></td>
  </tr> -->
  <!-- <tr>
    <td>GARF-Pro-Eggshell</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">PTv3-E</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge">GARF-Pro-EAO-FM-Eggshell</a></td>
    <td>fine-tuned on the eggshell subset of our Fractuna dataset</a></td>
  </tr>
  <tr>
    <td>GARF-Pro-Bone</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">PTv3-E</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge">GARF-Pro-EAO-FM-Bone</a></td>
    <td>fine-tuned on the bone subset of our Fractuna dataset</a></td>
  </tr>
  <tr>
    <td>GARF-Pro-Lithics</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">PTv3-E</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge">GARF-Pro-EAO-FM-Lithics</a></td>
    <td>fine-tuned on the lithics subset of our Fractuna dataset</a></td>
  </tr> -->
  <!-- <tr>
    <td>GARF-Ultra</td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-pt">PTv3-EAO</a></td>
    <td><a href="https://huggingface.co/IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge">GARF-Ultra-EAO-FM</a></td>
    <td>large-scale trained on everyday+artifact+other for both backbone and Flow-matching (cost most time!)</a></td>
  </tr> -->
</table>



## ✅ Evaluation Performance

## 🙋 FAQs


## Citation

## License

## Acknowledgement
 We gratefully acknowledge the Physical Anthropology Unit, Universidad Complutense de Madrid for providing access to the human skeletons under their curation. This work was supported in part through NSF grants 2152565, 2238968, 2322242, and 2426993, and the NYU IT High Performance Computing resources, services, and staff expertise. 


  <!-- ```bibtex
    @inproceedings{li2025garf,
    title={VGGT: Visual Geometry Grounded Transformer},
    author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2025}
    }
``` -->
