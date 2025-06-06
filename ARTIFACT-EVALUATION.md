# Artifact Appendix

Paper title: Robust and Efficient Watermarking of Large Language Models Using Error Correction Codes

Artifacts HotCRP Id: 3

Requested Badge: Functional

## Description
This artifact contains the necessary components to reproduce the experimental results in the paper, *Robust and Efficient Watermarking of Large Language Models Using Error Correction Codes*.
It includes the source code for the proposed watermarking scheme, evaluation scripts, and the fine-tuned model weights used for robustness evaluation.
Detailed instructions are provided for setup and for running the scripts to reproduce the results presented in the paper.
The artifact is publicly available on GitHub under the MIT license.

### Security/Privacy Issues and Ethical Concerns
To the best of our knowledge, this artifact poses no security, privacy, or ethical risks.

- **Security/Privacy**: We provide a self-contained Docker image to evaluate the artifact, requiring no root privileges or system modifications of the host system. The scripts simulating "attacks" are safe data-processing operations that pose no threat to the execution environment.
- **Ethical Concerns**: Fine-tuned model weights were obtained using the WikiText-2 dataset, which contains no PII. A successful watermark removal attack's only consequence is the inability to extract the watermark, which causes no broader harm.

## Basic Requirements

### Hardware Requirements
Evaluating this artifact requires *a GPU with at least 8 GB of VRAM*.
This is necessary for efficiently embedding and extracting watermarks from the large language models, as these operations are computationally intensive.
In addition, *at least 32 GB of RAM* is recommended to handle the large models and datasets involved in the evaluation.

### Software Requirements
The artifact is designed to be run within a Docker container, which encapsulates all necessary software dependencies and ensures a consistent environment.
No proprietary software or datasets requiring special access are used.
To run the Docker container, the reviewers *need to have Docker installed* on their system.
Alternatively, we provide detailed instructions in the README file for setting up the environment manually without Docker.
For reference, the following software components are included in the Docker image:

- Ubuntu 22.04 (amd64 architecture)
- Miniforge for managing the Python environment
- GCC and GMP library for building the shared library
- SageMath 10.4 and Python 3.11 for running the source code

### Estimated Time and Storage Consumption
Completely reproducing the results in the paper is expected to take more than 250 GB of disk space and about one week of computation time.
This is because the evaluation involves eight different large language models, with size ranging from 1B to 13B parameters, and each model requires substantial time for evaluating the utility, robustness, and efficiency of the watermarking scheme.
In addition, the corruption probability estimation for the watermark extraction process requires running more than a million simulations, which is computationally intensive.

Therefore, we prepare simplified experiments that can be finished in *about one hour* and require *less than 30 GB of disk space* to achieve the "Functional" badge.
The simplified experiments only use the 1B large language model to evaluate the utility, robustness, and efficiency of the watermarking scheme.
The number of simulations for estimating the corruption probability is also reduced, but it should be sufficient to validate the security of the watermarking scheme.

## Environment 

### Accessibility
The artifact is publicly available on the GitHub repository: https://github.com/luan-xiaokun/permumark.
The version used for evaluation is tagged as `v1.0.0`.
The associated fine-tuned model weights and the Docker image are also available in the repository.

### Set up the environment
We list three ways to set up the environment for evaluating the artifact: using the provided Docker image, building the Docker image locally, or setting up the environment manually.

#### Using the Provided Docker Image
We recommend using the provided Docker image to set up the environment for evaluating the artifact.
The Docker image can be pulled from the GitHub Container Registry with the following command:

```bash
docker pull ghcr.io/luan-xiaokun/permumark/permumark-eval:latest
```

To run the Docker container, you can use the following command:

```bash
docker run --gpus all -it --rm --gpus all \
  ghcr.io/luan-xiaokun/permumark/permumark-eval:latest
  # or just `permumark-eval:latest` if you build it locally
```

#### Building the Docker Image Locally
Alternatively, you can clone the artifact repository and build the Docker image locally with the following commands (takes about 40 minutes to build):

```bash
git clone git@github.com:luan-xiaokun/permumark.git
cd permumark
docker build -t permumark-eval:latest .
```

#### Manually Setting Up the Environment
As a last resort, you can also set up the environment manually if you do not have Docker installed.
In this case, please follow the detailed instructions in the README file.

### Testing the Environment
The artifact repository includes a simple test script to verify that the environment is set up correctly.
You can run the following command to execute the test script:
```bash
python envtest.py
# expected output (take about 1 minute):
# Extracted identity matches: True
```

This script imports the package `permumark` and inserts a watermark into the 1B Llama model.
It then extracts the watermark and checks if the extracted identity matches the inserted one.

## Artifact Evaluation

### Main Results and Claims
The proposed watermarking method achieves high model utility, high probability of detecting permutation corruption, high robustness against various modifications and attacks, and high efficiency in watermarking insertion and extraction.
Below, we summarize the main results and claims of the paper, which are supported by the experiments described in the next section.

#### Main Result 1: High Model Utility
Generation quality of watermarked models remains largely unchanged compared to the original models, with only a slight increase in perplexity.
This is demonstrated in Section 7.2 and supported by Experiment 1.

#### Main Result 2: High Probability of Detecting Permutation Corruption
The watermarking scheme can effectively detect permutation corruption with a high probability under different attack settings.
The probability of failing to detect corruption is estimated to be less than 1e-4, which is demonstrated in Section 7.3 and supported by Experiment 2.

#### Main Result 3: High Robustness Against Model Modifications
The proposed watermarking scheme is robust against various model modifications, including quantization, pruning, and fine-tuning.
The inserted watermark remains intact and can be extracted after these modifications, as demonstrated in Section 7.4.1 and supported by Experiment 3.

#### Main Result 4: High Robustness Against Watermark Obfuscation
The proposed watermarking scheme is robust against watermark obfuscation attacks based on functional invariant transformations.
The watermark can still be extracted after such attacks, as demonstrated in Section 7.4.2 and supported by Experiment 4.

#### Main Result 5: High Resistance to Watermark Removal and Forgery
The proposed watermarking scheme can defend against watermark removal if the number of corrupted permutations is within the correction capability of the error correction code.
Even if it exceeds the correction capability, a watermark forgery attack is unlikely to succeed, as demonstrated in Section 7.4.3 and supported by Experiment 5.

#### Main Result 6: High Efficiency of Watermarking Insertion and Extraction
The watermark insertion and extraction processes are efficient.
This is demonstrated in Section 7.5 and supported by Experiment 6.

### Experiments 
In this section, we describe the experiments to support the main results and claims.
All commands listed below should be executed in the root directory of the artifact repository.
Please note that the actual computation time may vary depending on the number of CPU cores available as we use multiprocessing to speed up the evaluation.

#### Experiment 1: Model Utility
This experiment evaluates the perplexity of the watermarked model and the original model without watermark.
It also evaluates the percentage of distorted tokens of the watermarked model compared to the original model.

- Expected results: The perplexity of the watermarked model should be similar to that of the original model, and the percentage of distorted tokens should be less than 8%.
- Estimated time: About 15 minutes.
- Supporting claim: This experiment supports Main Result 1, demonstrating that the watermarking scheme does not degrade model utility.
- Execution command: as follows.

```bash
# change the batch size according to your GPU memory capacity
python evaluation/evaluation.py utility llama --size 1b --batch_size 4
```
Please note that the evaluation result will be different from the one presented in the paper because we are using a different 1B model in the Docker image (the Llama-3.2-1B-Instruct model released by unsloth, instead of the original Llama-3.2-1B model released by meta-llama, which is more difficult to include in a Docker image).

#### Experiment 2: Probability of Detecting Permutation Corruption
This experiment estimates the probability of failing to detect permutation corruption during watermark extraction under different attack settings.
It runs a large number of simulations to estimate the probability.
The result presented in the paper is based on 100,000,000 (100 million) simulations for each configuration, which is computationally intensive.
For simplicity, we reduce the number of simulations to 1,000,000 (one million) and use a default configuration in this experiment.

- Expected results: The estimated probability of failing to detect corruption should be less than 1e-4.
- Estimated time: About 10 minutes.
- Supporting claim: This experiment supports Main Result 2, demonstrating that the watermarking scheme can effectively detect permutation corruption with high probability.
- Execution command: as follows.

```bash
# adjust the number of simulations according to your computational resources
python evaluation/probability_estimation.py corruption --num 1000000
```

We also provide the script to plot the figure presented in the paper (Figure 2), which can be run with `python evaluation/plots/probability.py`.

#### Experiment 3: Robustness Against Model Modifications 
This experiment evaluates the robustness of the watermarking scheme against various model modifications, including quantization, pruning, and fine-tuning.
The paper evaluates six large language models against these modifications under twelve different settings.
For simplicity, we only evaluate the 1B model in this artifact against six different settings, including the 4-bit quantization setting, the 0.5 sparsity Wanda pruning setting, and four different volumes of fine-tuning data.

- Expected results: The watermark should remain intact and can be extracted after the modifications.
- Estimated time: About 15 minutes.
- Supporting claim: This experiment supports Main Result 3, demonstrating that the watermarking scheme is robust against various model modifications.
- Execution commands: as follows.

```bash
# quantization
python evaluation/evaluation.py robustness llama --size 1b --modification quantization --quant_bits 4 
# pruning 
python evaluation/evaluation.py robustness llama --size 1b --modification pruning --pruning_method wanda
# fine-tuning
python evaluation/evaluation.py robustness llama --size 1b --modification finetune
```

Please note that evaluating the robustness against fine-tuning directly loads the fine-tuned model weights located in the directory `models/finetune` instead of fine-tuning the watermarked model on the fly.
If you are not using the provided Docker image, you need to download the fine-tuned model weights via this [link](https://github.com/luan-xiaokun/permumark/releases/download/PoPETs-Artifact/finetune-weights.tar.gz) and extract them to the right place.

#### Experiment 4: Robustness Against Watermark Obfuscation
This experiment evaluates the robustness of the watermarking scheme against watermark obfuscation attacks based on functional invariant transformations.
Specifically, we enable random weight permutations, vector scaling, and invertible matrix multiplication in this experiment.


- Expected results: The attack should fail and there should be no undetected corruption with a very high probability.
- Estimated time: About 1 minutes.
- Supporting claim: This experiment supports Main Result 4, demonstrating that the watermarking scheme is robust against watermark obfuscation attacks based on functional invariant transformations.
- Execution command: as follows.

```bash
python evaluation/evaluation.py security llama --size 1b --scale_attack --inv_attack
```

Please note that the result of this experiment is expected to report zero undetected corruption with a probability higher than 99.99%.
Therefore, if the result shows any undetected corruption, it indicates that you are getting really (un)lucky, but the attack should still fail as the error correction code can correct it.
(You can rerun the experiment in such a case to get a more reliable result, but I would personally strongly recommend you to buy a lottery ticket instead.)

#### Experiment 5: Resistance to Watermark Removal and Forgery
This experiment evaluates the resistance of the watermarking scheme to watermark removal and forgery attacks.
Specifically, it simulates an adversary who attempts to remove the watermark by corrupting a certain number of permutations in the model.
Any corruption within the correction capability of the error correction code can be detected and corrected during watermark extraction.
If the number of corrupted permutations exceeds the correction capability, the watermark may be lost, but the adversary cannot forge a new watermark with a reasonable probability.
For simplicity, we only demonstrate the case where the adversary tries to forge the watermark by corrupting all inserted permutations in the model.

- Expected results: There should be no successful watermark removal and no undetected corruption with a very high probability.
- Estimated time: About 5 minutes.
- Supporting claim: This experiment supports Main Result 5, demonstrating that the watermarking scheme has a high resistance to watermark removal and forgery.
- Execution command: as follows.

```bash
# adjust the number of simulations according to your computational resources
python evaluation/probability_estimation.py forgery --num 2000
```

Similarly, if the result differ from the one presented in the paper, it indicates that you are getting really (un)lucky.

#### Experiment 6: Efficiency of Watermarking Insertion and Extraction
This experiment evaluates the efficiency of the watermarking insertion and extraction processes.
It measures the time taken to insert and extract the watermark from the model.
The script will additionally report the time taken for solving the linear assignment problem during the extraction process.
For simplicity, we only evaluate the 1B model in this experiment.

- Expected results: The time taken for watermark insertion and extraction should be within 5 seconds and 15 seconds on a workstation-level platform, respectively.
- Estimated time: About 3 minutes.
- Supporting claim: This experiment supports Main Result 6, demonstrating that the watermarking insertion and extraction processes are efficient.
- Execution command: as follows.

```bash
# use `--repeat REPEAT` to change the number of runs for averaging the time
python evaluation/evaluation.py efficiency llama --size 1b
```
The actual evaluation result may differ from the one presented in the paper and the estimated one above due to different hardware.
For simplicity, we only evaluate the time taken for inserting and extracting the proposed watermark on the 1B model, without evaluating the compared baseline method.

## Limitations
All tables and results presented in the paper are reproducible with the provided artifact given enough computational resources.
However, some results may vary due to different hardware (e.g., efficiency evaluation results) and randomness (e.g., estimated probability) in the evaluation process.
We have clearly stated the platform we used for evaluation in the paper for reference.

## Notes on Reusability (Only for Functional and Reproduced badges)
We prepared the source code as a reusable Python package that can be easily installed and used in other projects.
The design of the package allows for easy application of the watermarking scheme to other large language models.
A submodule of the package is for efficiently computing derangements, following a [state-of-the-art algorithm](https://doi.org/10.1016/j.dam.2016.10.001).
It is written in C and has a Python interface, which can be reused in other research projects that require derangement computations.
In addition, the source code also includes an implementation of the [baseline method](https://inria.hal.science/hal-04361026) we compared against, which can be used as a reference for other watermarking schemes.
