from transformers import PreTrainedModel

# PermuMark

PermuMark is an error correction enhanced permutation watermark for transformer models.
As a white-box training-free watermarking method, the insertion and extraction of watermarks is efficient and does not require GPU (of course it is good to have one to accelerate the extraction process).
Moreover, PermuMark is robust under various model modification techniques, including fine-tuning, quantization, and pruning.
The watermark can be preserved and successfully extracted even under adaptive attacks, as long as the attacks are within the error correction capability of PermuMark.
Last but not least, PermuMark has minimal impact on model performance.

## Requirements & Build

### TL;DR

1. install libgmp using `sudo apt-get install libgmp-dev`
2. prepare a Python3.11 environment `conda create -n permumark python=3.11 && conda activate permumark`
3. install SageMath `conda install sage=10.4`
4. build shared library `./build.sh`
5. install the package `pip install -e .`
6. (Optional) install AutoGPTQ for evaluating robustness under quantization

### Non-Python Library

- gcc (or any other C compiler)
- [GMP](https://gmplib.org/) (e.g., through `sudo apt-get install libgmp-dev`)
- [SageMath](https://www.sagemath.org/) 10.4 (e.g., through `conda install sage=10.4`)

The ranking and unranking of derangement are implemented in C and exposed to Python through a shared library (DLL should
also work).
To build the shared library, simply run `./build.sh` under the project directory, it will produce a shared library under
`permumark/derangement`.

### Python Packages

PermuMark is developed on Python3.11, and it should also work for Python>=3.10.
The following dependencies can be installed via `pip install -e .`

- datasets==3.2.0
- torch==2.2.1
- scipy==1.14.1
- sympy==1.13.3
- transformers==4.47.0

#### AutoGPTQ for Quantization

To evaluate the robustness of PermuMark under GPTQ quantization, [`auto_gptq`](https://github.com/AutoGPTQ/AutoGPTQ) is
required.
We recommend [install from source](https://github.com/AutoGPTQ/AutoGPTQ?tab=readme-ov-file#install-from-source) to set
up the environment.
PermuMark was evaluated under PyTorch 2.2.1+cu118 version.

## Usage

### Watermark Insertion

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

### Watermark Extraction

Suppose the original model without watermark is `source`, and the watermarked model is `model`,
the identity (and also watermark) can be extracted using `extract_watermark` as follows.

```python
extract_res = pw.extract_watermark(source, model)
print("Extracted identity:", extract_res.identity)
print("Extracted watermark:", extract_res.watermark)
```

The extracted identity and watermark can be accessed through the extraction result `extract_res`.
Similarly, inserted identity and watermark can be accessed through the insertion result `insert_res`.

### Watermark Configuration

`PermutationWatermark` accepts two arguments: `max_corrupt_prob` and `total_id_num`.

- `max_corrupt_prob`: Maximum probability of undetectable adversary corruption that the model owner can tolerate.
  Defaults to 1e-4 (i.e., 0.01%), which means that when the adversary corrupts a permutation at certain layer, the model
  owner can detect it with more than 99.99% probability when extracting the watermark.
- `total_id_num`: The number of different model identifiers that the model owner wants to manage.
  Defaults to 10,000,000, meaning that the model owner can distribute at most 10M models with different identifiers.

## Evaluation

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

Evaluation results presented in the report can be reproduced by running scripts placed under `scripts` folder.
