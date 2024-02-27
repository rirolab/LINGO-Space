<div align="center">
    <h1> <a>LINGO-Space: Language-Conditioned Incremental Grounding for Space</a></h1>

<p align="center">
  <a href="https://lingo-space.github.io">Project Page</a> •
  <a href="https://arxiv.org/abs/2402.01183">Paper</a> •
  <a href="https://github.com/rirolab/LINGO-Space">Code</a> •
  <a href="#bibtex">BibTex</a>
</p>

</div>


LINGO-Space is a novel probabilistic space-grounding methodology that accurately identifies a probabilistic distribution of space being referred to and incrementally updates it, given subsequent referring expressions leveraging configurable polar distributions.

<img width="1194" alt="pipeline" src="./assets/images/pipeline.jpg">
</details>


## Installation
Clone this repository:
```
git clone --recurse-submodules https://github.com/rirolab/LINGO-Space.git
```

Create a conda environment:
```
conda create -n lingo_space python=3.8
conda activate lingo_space
```

Install [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/), [CLIP](https://github.com/openai/CLIP) and other dependencies:
```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

Install [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO):
```
export CUDA_HOME=/path/to/cuda-11.7
cd sgg/GroundingDINO/
pip install -e .
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Usually, you can set `CUDA_HOME` as:
```
export CUDA_HOME=/usr/local/cuda-11.7
```

Downlaod LINGO-Space's dataset:
```
sh scripts/download_dataset.sh
```


## Usage
TODO: You can check our demo on [Colab]()!

### Running the training code
First, you need to set "OPENAI_API_KEY" to use the semantic parser:
```
export OPENAI_API_KEY='YOUR API KEY'
```
Please, refer [OpenAI](https://openai.com/) to get an API key.

Then, run the train code:
```
python train.py dataset.type=composite
```

You can replace `dataset.type` as you want, e.g., 'close-seen-colors'.

Refer [cfg](./cfg) directory for detailed configurations.

### Running the test code
Run the test code:
```
python test.py --dataset_type composite --ndemos_test 50 
```


### Training with a custom dataset
Unfortunately, we are not releasing the dataset generation code because of licensing issues.
The dataset generation code is built on [ebmplanner](https://github.com/ayushjain1144/ebmplanner) and [cliport](https://github.com/cliport/cliport), so please refer those original repositories.

Also, for those who want to train LINGO-Space with a custom dataset, we specify the format of dataset below.

#### Structure
```
data
├── {dataset_type}-train
│   ├── images
│   │   ├── 000000-0
│   │   │   ├── 0.png
│   │   │   └── 0.png
│   │   └── ...
│   ├── info
│   │   ├── 000000-0.pkl
│   │   └── ...
├── {dataset_type}-val
│   └── ...
└── {dataset_type}-test
    └── ...
```

#### Data that should be in the "info"

Most of them are same as [ebmplanner](https://github.com/ayushjain1144/ebmplanner) and [cliport](https://github.com/cliport/cliport), but we need additional data for LINGO-Space:

* "rel_ids": a list of ids of objects to be the source
* "ref_ids": a list of ids of reference objects of the target
* "relations": a list of relational predicates (e.g., close)
* "graph": a list of (node_id1, edge, node_id2) (e.g., `[(7, 'near', 6), (6, 'near', 7)]`). It represents a scene graph of the environment.


## Troubleshooting
If you are having trouble installing opencv, try the following:
```
sudo apt-get install -y libglib2.0-0
sudo apt-get install -y libsm6
```

## Acknowledgement
We thank to open source repositories: [GraphGPS](https://github.com/rampasek/GraphGPS), [ebmplanner](https://github.com/ayushjain1144/ebmplanner), [cliport](https://github.com/cliport/cliport), and [paragon](https://github.com/1989Ryan/paragon).

## BibTex
```
@article{kim2024lingo,
  title={LINGO-Space: Language-Conditioned Incremental Grounding for Space},
  author={Kim, Dohyun and Oh, Nayoung and Hwang, Deokmin and Park, Daehyung},
  journal={arXiv preprint arXiv:2402.01183},
  year={2024}
}
``` 