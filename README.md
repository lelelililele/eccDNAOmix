# EccDNA-explorer  

## Installation  

You can install just the base python(v3.8) packages, include: 
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
### DNA SNV, SV, 5mC_5hmC and 6mA:  

At this step, the following input files need to be prepared in advance:    

- SNV mutation allele frequency file `N1_mutation_AF` (detailed procedures can be found in `SNV_AF_process`)  
- SV VCF file `N1_SV.vcf`  
- 5mC/5hmC and 6mA pileup files `N1_pileupALL.bed.gz` and `N1_6mA_pileup.bed.gz`
  
DNA SNV:  
```bash
python SNVAF_to_input.py -b N1_eccDNA.sort_extended.bed -a N1_mutation_AF -o N1_SNP_AFoutput.npz
```
DNA SV:
```bash
python SV_to_onehot.py -b N1_eccDNA.sort_extended.bed -v N1_SV.vcf -o N1_extended_SV.npz
```
DNA 5mc_5hmc:
```bash
python DNA_5mc_5hmc.py N1_eccDNA.sort_extended.bed N1_pileupALL.bed.gz N1_5mc5hmc_output.npz
```
DNA 6mA:
```bash
python DNA_6mA.py -a N1_eccDNA.sort_extended.bed -b N1_6mA_pileup.bed.gz -o N1_6mA_output.npz
```

### RNA m6A TPM:
At this step, the following input files need to be prepared in advance:  

- RNA modification data `N1_RNA_mod`  
- Transcript count matrix `N1_transcript_counts.tsv`  

For TPM (Transcripts Per Million) normalization, please refer to the preparation script: `eccDNA_TPM.pbs`

RNA m6A:  
```bash
python m6A_to_input.py -b N1_eccDNA.sort_extended.bed -a N1_RNA_mod -o N1_RNA_modoutput.npz  
```
RNA TPM:  
```bash
python RNA_TPM.py N1_eccDNA.sort_extended.bed N1_transcript_counts.tsv N1_eccDNA_TPM.npz
```


## Step2: Data combine  

## Step3: Model training  

