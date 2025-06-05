# Artifact for PoPETs 2025 Paper: Robust and Efficient Watermarking of Large Language Models Using Error Correction Codes

This artifact includes the source code, evaluation scripts, fine-tuned weights, and raw evaluation results for the paper *Robust and Efficient Watermarking of Large Language Models Using Error Correction Codes* accepted at PoPETs 2025.

We describe the artifact in detail below, including how to install and use the library, how to set up the environment for evaluation, and how to reproduce the evaluation results presented in the paper.

## Source Code: PermuMark

The watermarking scheme proposed in the paper is implemented in the `permumark` Python package, called *PermuMark* and located in the `permumark` directory.
PermuMark is an error correction enhanced permutation watermark for transformer models.
As a white-box training-free watermarking method, the insertion and extraction of watermarks is efficient and does not require GPU (of course it is good to have one to accelerate the extraction process).
Moreover, PermuMark is robust under various model modification techniques, including fine-tuning, quantization, and pruning.
The watermark can be preserved and successfully extracted even under adaptive attacks, as long as the attacks are within the error correction capability of PermuMark.
Last but not least, PermuMark has minimal impact on model performance.

Below, we provide a detailed description of the requirements and how to build the package.

### Requirements

We need the following tools and libraries to build and use PermuMark:

- `conda` for creating a Python environment and installing SageMath
- `gcc` and `libgmp-dev` for building the shared library

For reference, the artifact was developed and tested on Ubuntu 22.04 with amd64 architecture.
To install the required libraries, you can run the following commands:

```bash
# we recommend using Miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
# make sure gcc and gmp are installed
sudo apt install build-essential
sudo apt install libgmp-dev
```

### Installation

First create a new environment for PermuMark using `conda`:

```bash
# you may need to first run `conda init` to set up your shell
conda create -n permumark python=3.11
conda activate permumark
# you can verify the path of the Python interpreter
which python
```

Then install [SageMath](https://www.sagemath.org/) through `conda` and build the shared library for derangement (commands below assume you are in the project root directory):

```bash
conda install sage=10.4
./scripts/build.sh
# a shared library should be built in permumark/derangement
```

To verify that SageMath is installed correctly, you can run `sage` in the terminal, and it should output something like this without warnings or errors:

```text
┌────────────────────────────────────────────────────────────────────┐
│ SageMath version 10.4, Release Date: 2024-07-19                    │
│ Using Python 3.11.12. Type "help()" for help.                      │
└────────────────────────────────────────────────────────────────────┘
sage: 
```

**NOTE**: You may need to pass `-c conda-forge` to `conda install` if you are not using Miniforge, which uses conda-forge as the default channel.

Finally, install the dependencies and the package using `pip`:

```bash
pip install -e .
```

The main dependencies include:

- datasets==3.2.0
- peft==0.14.0
- tokenizers==0.21.1
- torch==2.2.1
- transformers==4.47.0

### Usage

**Watermark Insertion**:
To insert permutation watermarks, first create a `PermutationWatermark` instance by passing the model config.
Then the watermark can be generated and inserted to the model within two lines.

```python
from transformers import AutoModelForCausalLM
from permumark import PermutationWatermark

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", trust_remote_code=True)
# create a PermutationWatermark instance by passing the model config
pw = PermutationWatermark(model.config)
# generate a random identity for the model
identity = pw.generate_random_identity()
# insert the watermark to the model
insertion_result = pw.insert_watermark(model, identity)
```

**Watermark Extraction**:
Suppose the original model without watermark is `source`, and the watermarked model is `model`,
the identity (and also watermark) can be extracted using `extract_watermark` as follows.

```python
extract_res = pw.extract_watermark(source, model)
print("Extracted identity:", extract_res.identity)
print("Extracted watermark:", extract_res.watermark)
```

The extracted identity and watermark can be accessed through the extraction result `extract_res`.
Similarly, inserted identity and watermark can be accessed through the insertion result `insert_res`.

**Watermark Configuration**:
`PermutationWatermark` accepts two arguments: `max_corrupt_prob` and `total_id_num`.

- `max_corrupt_prob`: Maximum probability of undetectable adversary corruption that the model owner can tolerate.
  Defaults to 1e-4 (i.e., 0.01%), which means that when the adversary corrupts a permutation at certain layer, the model
  owner can detect it with more than 99.99% probability when extracting the watermark.
- `total_id_num`: The number of different model identifiers that the model owner wants to manage.
  Defaults to 10,000,000, meaning that the model owner can distribute at most 10M models with different identifiers.

## Evaluation Scripts and Results

We provide evaluation scripts to assess the utility, robustness, and efficiency of the proposed watermarking scheme.
The evaluation entrance and utilities are located in the `evaluation` directory.
Scripts for setting up the evaluation environment (e.g., downloading models and datasets) and reproducing the results presented in the paper are placed under the `scripts` directory.

Below, we describe how to set up the evaluation environment and run the evaluation scripts.
Part of the evaluation code is adapted from [Wanda](https://github.com/locuslab/wanda) and [SparseGPT](https://github.com/IST-DASLab/sparsegpt).


### Evaluation Environment Setup

**Hardware Requirements**:
At least 128GB of RAM and an GPU with at least 24GB of VRAM for evaluating large models.
Downloading the models and datasets requires at least 250GB of disk space.

**Download Models and Datasets**
We provide a script `scripts/download.sh` for downloading the models and datasets used in the evaluation.
To use the script, first install `huggingface_hub[cli]` and then login to Huggingface using your account token (for downloading Llama models):

```bash
pip install huggingface_hub[cli]
huggingface-cli login
# enter your Huggingface account token when prompted
./scripts/download.sh
```

**Dependency Installation**:
To evaluate the robustness of the proposed watermarking scheme against GPTQ quantization, [`auto_gptq`](https://github.com/AutoGPTQ/AutoGPTQ) is
required.
We recommend [install from source](https://github.com/AutoGPTQ/AutoGPTQ?tab=readme-ov-file#install-from-source) to set
up the environment:

```bash
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
pip install numpy gekko pandas
pip install -vvv --no-build-isolation -e .
```

This requires `nvcc` to be installed, which is usually included in the CUDA toolkit.
An alternative installation method is to set `BUILD_CUDA_EXT=0` to fall back to a pure Python implementation, but this will be much slower.
PermuMark was evaluated under PyTorch 2.2.1+cu118 version without using the CUDA extension (in which case warnings about CUDA extension are expected).

### Reproducing Evaluation Results

**Usage**:
The main evaluation entrance is placed at `evaluation/evaluation.py`.

```
usage: evaluation.py [-h] [--batch_size BATCH_SIZE] [--inv_attack]
                     [--max_corrupt_prob MAX_CORRUPT_PROB]
                     [--modification MODIFICATION] [--perm_budget PERM_BUDGET]
                     [--perm_type PERM_TYPE] [--quant_bits QUANT_BITS]
                     [--repeat REPEAT] [--scale_attack] [--simulate]
                     [--size SIZE] [--torch_dtype TORCH_DTYPE]
                     [--total_id_num TOTAL_ID_NUM] [-v]
                     {utility,robustness,efficiency,security} model_type
```

For example:

```bash
# evaluate efficiency
python evaluation/evaluation.py efficiency llama --size 7b --repeat 3 --verbose
# evaluate robustness
python evaluation/evaluation.py robustness qwen --size 7b --modification finetune
# evaluate utility
python evaluation/evaluation.py utility gemma --size 7b --batch_size 4
# evaluate security
python evaluation/evaluation.py security llama --size 8b --perm_budget 60 --perm_type random
```

For detailed usage, refer to the description in `ARTIFACT-EVALUATION.md`.

**Plotting Figure**:
Run `python evaluation/plots/probability.py` to plot the estimated probability of undetectable corruption for configurations.

**Evaluation Results**:
We provide the log files of the evaluation results in the `evaluation/logs` directory.

## License

This artifact is released under the MIT License.
