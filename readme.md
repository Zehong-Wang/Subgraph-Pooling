# Subgraph Pooling: Tackling Negative Transfer on Graphs

## Getting Started

### Setup Environment

We use conda for environment setup. Please run the bash as 
```
conda create -y -n SP python=3.8
conda activate SP

pip install -r requirements.txt
```

Here a conda environment named `SP` and install relevant requirements from `requirements.txt`. Be sure to activate the environment via `conda activate SP` before running experiments as described. 

### Dataset Preparation

We include six datasets in this paper. Please download from the following links and put them under `data/` like
```
.
|- params
|
|- data
|   |- twitch
|   |- airports 
|   |- elliptic
|   |- ...
```

- **Citation Network** (`ACM` and `DBLP`): Download the `.zip` file from [here](https://drive.google.com/file/d/1vRKjBbu5OBlxUBoFaz48PY0_UfwFhg-Z/view?usp=sharing) and decompress under `data` folder. More details [here](https://dl.acm.org/doi/abs/10.1145/3366423.3380219?casa_token=TaeY-qwyU_wAAAAA:Jf3P4aYEiE0wjHwj2XJPKkddrxAUL0Qfx6sH3nuKURwVK79nZTWU4HSejxfET5aHrjiiS0Cwq1IhvA).
  
- **Airport Network** (`USA`, `Brazil`, and `Europe`): These datasets will be automatically downloaded when running the code. More details [here](https://arxiv.org/abs/1704.03165).

- **Twitch Network** (`DE`, `EN`, `ES`, `FR`, `PT`, `RU`): The dataset are collected in different countries, which have varying sizes and distributions. These graphs will be automatically downloaded when running the code. More details [here](https://arxiv.org/abs/1909.13021). 

- **OGB-Arxiv**: This is a citation network containing papers published from 2005 to 2020. The dataset has two settings, the first is to evaluate the temporal distribution shift (which will be downloaded automatically), and the second is to evaluate the degree shift (with dataset [here](https://drive.google.com/file/d/17r1x42SW0KQE7uh7W1VubPXscv2EQkHl/view?usp=sharing)). Please decompress the file under `data` folder. More details are presented in [OGB Benchmark](https://ogb.stanford.edu/) and [GOOD](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html). 

- **Elliptic Network**: A Bitcoin transaction network. Download [here](https://drive.google.com/file/d/1Mv3ufeaJDbop5VubUmBNKgn_ueL7hTBs/view?usp=sharing) and decompress under `data` folder. More details [here](https://arxiv.org/abs/2202.02466). 

- **Facebook100 Network**: The dataset contains 14 graphs collected from Facebook. Please download [here](https://drive.google.com/file/d/1lX2EfZlXCV-Q1lM6gekGrNGqA0HejNmZ/view?usp=sharing) and decompress under `data` folder. More details [here](https://arxiv.org/abs/2202.02466).

### Usage

To quickly run the model, please run `main.py` by specifying the experiment setting. Here is an example. 
```
python main.py --use_params --source_target acm_dblp --backbone gcn --sampling k_hop --ft_last_layer
```
Note that we provide three transfer learning settings, including `freeze` (do not fine-tune on the target graph), `ft_last_layer` (fine-tune the last layer), and `ft_whole_model` (fine-tune the whole model). You can jointly run these settings, like 
```
python main.py --use_params --source_target acm_dblp --freeze --ft_last_layer --ft_whole_model
```
In the following, we give a simpler method for running the code. 

## Reproducibility

To ensure reproducibility of the paper, we provide the detailed hyper-parameters under `params` folder. One simple method is to run the bash script under `script` folder, like

```
bash script/run.sh DATASET BACKBONE SAMPLING
```
where the first term indicates the `dataset`, the second term indicates the used `backbone`, and the last one is the subgraph `sampling` method. Please refer to [script/readme.md](script/readme.md) for more details. 

Here are some examples. 

```
bash script/run.sh acm_dblp gcn k_hop
bash script/run.sh dblp_acm gcn k_hop
bash script/run.sh arxiv_1_arxiv_5 gcn rw
bash script/run.sh arxiv_3_arxiv_5 gcn rw
```

To extend Subgraph Pooling to your own model, one simple method is to implement your own model under `model.py`. 

## Contact Us

Please open an issue or contact `zwang43@nd.edu` if you have questions. 