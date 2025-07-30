# ​​eccDNAOmix​​  
## Introduction
Extrachromosomal circular DNA (eccDNA) represents a unique class of circular DNA molecules derived from chromosomes that are closely associated with oncogene amplification. However, eccDNAs exhibit significant heterogeneity among individuals, making it challenging to characterize them using uniform data features. In this study, we developed a multimodal deep learning framework for predicting and characterizing eccDNAs by performing multi-omics sequencing on colorectal cancer tumor tissues and paired adjacent normal tissues. Integrating our former established eccDNA sequencing pipeline with multi-dimensional omics data including DNA sequences, single nucleotide variants/insertions-deletions (SNV/InDel), structural variations (SV), DNA/RNA modifications, and RNA expression profiles, we constructed an adaptive gated fusion model that dynamically weights the contributions of each modality while employing a masking strategy to filter out non-biological noise. The model demonstrated well performance in validation set (AUC=0.823), with the DNA sequence modality showing the highest predictive contribution (AUC=0.756).   
![figure1](https://github.com/user-attachments/assets/9e3de0cc-d907-499a-a81c-0ae1ef853adf)

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
conda create -n eccDNAOmix python=3.8.5
```

2. Activate conda  

```bash  
conda activate eccDNAOmix
```

3. Install python packages  

```bash
pip install numpy==1.24.3 pandas==2.0.3 scipy==1.10.1 biopython==1.83 scikit-learn==1.3.2 xgboost==0.90 umap-learn==0.5.7 torch==2.4.1 matplotlib==3.7.5
```  

The software [bedtools](https://bedtools.readthedocs.io/en/latest/index.html) and [human hg38 reference fasta](https://github.com/broadinstitute/gatk/blob/master/src/test/resources/large/Homo_sapiens_assembly38.fasta.gz) were also needed.   

## If you want to directly use the pre-trained human model, you can skip to ​​[Step 4](### Step4: eccDNA predict).
## Step1: Data preparation  
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


## Step2: Data combination    
After completing Step 1 and obtaining separate .npz files for each sample, we merged these files into one combined .npz file.  
### DNA sequence One-hot:  
```bash  
python eccDNA_onehot_npz_process.py -i N1_eccDNA_onehot.npz N2_eccDNA_onehot.npz N3_eccDNA_onehot.npz N4_eccDNA_onehot.npz N5_eccDNA_onehot.npz -o N1-N5_eccDNA_onehot_merged_arrays.npz  
```  
### DNA SNV:  
```bash  
python eccDNA_SNP_AFoutput.py -i N1_SNP_AFoutput.npz N2_SNP_AFoutput.npz N3_SNP_AFoutput.npz N4_SNP_AFoutput.npz N5_SNP_AFoutput.npz -o N1-N5_SNP_AFoutput.npz  
```  
### DNA SV:  
```bash  
python eccDNA_extended_SV.py -i N1_extended_SV.npz N2_extended_SV.npz N3_extended_SV.npz N4_extended_SV.npz N5_extended_SV.npz -o N1-N5_extended_SV.npz  
```  
### DNA 5mc_5hmc:  
```bash  
python eccDNA_5mc5hmc_output.py -i N1_5mc5hmc_output.npz N2_5mc5hmc_output.npz N3_5mc5hmc_output.npz N4_5mc5hmc_output.npz N5_5mc5hmc_output.npz -o N1-N5_eccDNA_5mc5hmc_output.npz  
```  
### DNA 6mA:  
```bash  
python eccDNA_6mA_output.py -i N1_6mA_output.npz N2_6mA_output.npz N3_6mA_output.npz N4_6mA_output.npz N5_6mA_output.npz -o N1-N5_6mA_output.npz  
```  
### RNA m6A:  
```bash  
python eccDNA_RNA_modoutput.py -i N1_RNA_modoutput.npz N2_RNA_modoutput.npz N3_RNA_modoutput.npz N4_RNA_modoutput.npz N5_RNA_modoutput.npz -o N1-N5_RNA_modoutput.npz  
```  
### RNA TPM:   
```bash  
python eccDNA_TPM.py -i N1_eccDNA_TPM.npz N2_eccDNA_TPM.npz N3_eccDNA_TPM.npz N4_eccDNA_TPM.npz N5_eccDNA_TPM.npz -o N1-N5_eccDNA_TPM.npz  
```

## Step3: Model training  
After preparing the input dataset, execute main.py to train multiple deep learning models:
```bash 
python main.py
```
![画图_3](https://github.com/user-attachments/assets/88d68d45-7656-4b71-8a5c-166be381babe)



All output files will be saved in the ./output/ directory.

## Step4: eccDNA predict
Users can directly utilize our trained model to predict whether DNA sequences can form eccDNA. Please note that input sequences should include 100bp upstream and downstream of the potential eccDNA start/end sites, with candidate sequences not exceeding 1000bp in length.  

```bash 
python eccDNAPredict.py -i file.txt -o eccDNA_output
```

```bash 
file.txt: 
ATCGATCGATGCTAGCTAGCTAGGCTAGCTAGCTAGGCTAGCTAGCTAGGCTAGCTAGCTAGATC...
TCGATGCTAGCTAGCTAGGCTAGCTAGCTAGGCTAGCTAGCTAGGCTAGCTAGCTAGATCGATCG...
```
Each line is an independent sequence.
![eccDNA](https://github.com/user-attachments/assets/2cc524dc-007b-4420-9a66-114150928aab)


The prediction file was like:  
|SequenceID|Sequence|Prediction|Probability|
|----------|--------|----------|-----------|
|1         |ATCG... |Non-eccDNA|0.0942     |  
|2         |TCGA... |eccDNA    |0.967      |  


