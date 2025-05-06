# EccDNA-explorer  

## Installation  

You can install just the base python(3.8) packages, include: 
- numpy
- pandas
- scipy
- biopython
- scikit-learn
- xgboost
- umap-learn
- torch
- matplotlib

We recommend creating the environment and installing it with conda:  

1. Create conda envireoment  
  
```bash
conda create -n eccDNA_explorer python=3.8.5
```

2. Activate conda  

```bash  
conda activate eccDNA_explorer
```

3. Install python packages  

```bash
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.10.1 biopython==1.83 scikit-learn==1.3.2 xgboost==0.90 umap-learn==0.5.7 torch==2.4.1 matplotlib==3.7.5
```  

The software [bedtools](https://bedtools.readthedocs.io/en/latest/index.html) and [human hg38 reference fasta](https://github.com/broadinstitute/gatk/blob/master/src/test/resources/large/Homo_sapiens_assembly38.fasta.gz) were also needed.   


## Step1: Data process  
In this step, we require an example file 'N1_eccDNA.sort.bed' to demonstrate the analysis pipeline. 

### DNA sequence One-hot:
Convert eccDNA positional information into one-hot encoded files (saved as NPZ format), while generating extended ±100bp files: `N1_eccDNA.sort_extended.bed` and `N1_eccDNA.sort_extended_sequences.fasta` for subsequent steps.
```bash
python bed_to_onehot.py N1_eccDNA.sort.bed Homo_sapiens_assembly38.fasta N1_eccDNA_onehot.npz
```
Generate negative control samples `N1_OPeccDNA.bed` by randomizing chromosomal coordinates:
```bash
bedtools shuffle -i N1_eccDNA.sort.bed -g hg38.chrom.sizesrescaffold -excl N1_eccDNA.sort.bed > N1_OPeccDNA.bed
bedtools sort -i N1_OPeccDNA.bed > N1_OP_eccDNA.sort.bed
```
Perform one-hot conversion for negative controls, simultaneously generating extended ±100bp BED and FASTA files:
```bash
python bed_to_onehot.py N1_OP_eccDNA.sort.bed Homo_sapiens_assembly38.fasta N1_eccDNA_onehotOP.npz
```
### DNA SNV SV 5mC_5hmC 6mA:
在这一步，需要提前准备好SNV的mutation_AF, SV的vcf file, 5mC_5hmC和6mA pileup file.
mutation_AF


## Step2: Data combine  

## Step3: Model training  

