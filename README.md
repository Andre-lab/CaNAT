# CaNAT
CaNAT : Codon from Amino Acid with a Non-Autoregressive Transformer

Predicts codons and associated confidence score from amino acid sequences.


## Installation
```bash
git clone https://github.com/Andre-lab/CaNAT.git
cd CaNAT

Create a Conda environment with all dependancies:

```bash
conda env create -f environment_full.yml -n canat

# Activate the environment
conda activate canat


## Usage / Inference

Run the inference code for a fasta file 'sequence.fasta'. Result will be stored in 'output_dir':

```bash

# Run the inference script
python network/scripts/CaNAT/inference.py -p network/scripts/CaNAT/parameters_CaNAT.pt -o your_output_dir -i sequence.fasta
Output is a file with the same number of columns as the input amino acid sequence. The first line lists all codons. Each cell contains the predicted value for a codon at a given position. Confidence scores are calculated by applying the softmax to each row.
