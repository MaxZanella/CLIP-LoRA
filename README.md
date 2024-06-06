# Low-Rank Few-Shot Adaptation of Vision-Language Models [CVPRW 2024]

The official implementation of [*Low-Rank Few-Shot Adaptation of Vision-Language Models*](https://arxiv.org/abs/2405.18541).

**Authors**:
[Maxime Zanella](https://scholar.google.com/citations?user=FIoE9YIAAAAJ&hl=fr&oi=ao),
[Ismail Ben Ayed](https://scholar.google.com/citations?user=29vyUccAAAAJ&hl=fr&oi=ao).

We present CLIP-LoRA, an easy-to-use few-shot method for Vision-Language Models with fixed hyperparameters for every task and every number of shots. This repository also aims at facilitating the usage of Low-Rank adapters (LoRA) in Vision-Language Models like CLIP.

<p align="center">
  <img src="peft2.jpg" alt="PEFT" width="300" height="250">
  <br>
  <em>Figure 1: Low-Rank Adaptation (LoRA) is easy to use and does not create any additional inference latency.</em>
</p>

Here is how to run the experiments:

1. [Installation](#installation) 
2. [Usage](#how-to-execute-CLIP-LoRA) 

A quick guide on how LoRA is implemented in this repository:

3. [LoRA in MultiheadAttention](#lora-in-multiheadattention)

Please consider supporting our work:

4. [Citation](#citation)
   

## Installation 

### Environment configuration

Our code requires an environment with PyTorch installed. If you don't have one, consider creating a Python environment with:
```bash
conda create -y --name CLIP-LoRA python=3.10.0
conda activate CLIP-LoRA
```
And install Pytorch for instance with:
```bash
pip3 install torch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2
```

### Datasets installation

Please follow [DATASETS.md](DATASETS.md) to install the datasets.

## How to execute CLIP-LoRA

Execute CLIP-LoRA on the ImageNet dataset with a random seed of 1 by entering the following command:

```bash
python main.py --root_path /path/to/your/data --dataset imagenet --seed 1
```

You can also exectute CLIP-LoRA on the 10 other datasets:

```bash
python main.py --root_path /path/to/your/data --dataset dataset_name --seed 1
```

You can optionally provide a save_path to save the LoRA modules, which can be reload easily with the --eval_only argument. The code will automatically check if your trained LoRA with the corresponding rank, alpha, encoder, params and position to ensure compatibility. The folder will be structured like that:
```
/your/save/path
└── backbone
    └── dataset
        └── Xshots
            ├── seedY
```

Here is the command line:
```bash
python main.py --root_path /path/to/your/data --dataset dataset_name --seed 1 --save_path /your/save/path --eval_only 
```

## LoRA in MultiheadAttention

The `PlainMultiheadAttentionLoRA` class in `loralib/layers.py` extends the standard PyTorch multi-head attention mechanism by incorporating Low-Rank Adaptation (LoRA). This class constructs explicit linear modules for each component of the attention mechanism—query (`q`), key (`k`), value (`v`), and output (`o`)—providing a structured and adaptable foundation for your experiments.

### Class Overview

`PlainMultiheadAttentionLoRA` takes an existing `nn.MultiheadAttention` module, replicates its configuration, and integrates LoRA linear modules.

### Key Features

- **Parameter Initialization:** The initialization process involves copying weights and biases from a pre-existing multi-head attention model. Each LoRA module (`q`, `k`, `v`, `o`) is adapted based on the specified requirements in the `enable_lora` list.
- **LoRA Integration:** The replacement of standard linear layers with `LinearLoRA` layers introduces low-rank matrices, which are parameterized by the rank of adaptation (`r`) and the scaling factor (`lora_alpha`).
- **Forward Pass:** The `forward_module` method manages the attention computation, incorporating optional dropout settings on the LoRA modules.

### Example Usage

The following snippet demonstrates how to initialize the `PlainMultiheadAttentionLoRA` with an existing multi-head attention module.

```python
from loralib.layers import PlainMultiheadAttentionLoRA

# Initialize with an existing MultiheadAttention module
existing_mha = nn.MultiheadAttention(embed_dim=512, num_heads=8)
lora_mha = PlainMultiheadAttentionLoRA(existing_mha, enable_lora=['q', 'k', 'v', 'o'], r=4, lora_alpha=2)
```

## Citation

If you find this project useful, please cite it as follows:

```bibtex
@article{zanella2024low,
  title={Low-Rank Few-Shot Adaptation of Vision-Language Models},
  author={Zanella, Maxime and Ayed, Ismail Ben},
  journal={arXiv preprint arXiv:2405.18541},
  year={2024}
}
```
## Acknowledgement

We express our gratitude to the [CoOp](https://github.com/KaiyangZhou/CoOp) and [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter) authors for their open-source contribution.


