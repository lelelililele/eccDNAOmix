import os

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, 'data')
NEG_PATH = os.path.join(DATA_DIR, 'negative_samples')
POS_PATH = os.path.join(DATA_DIR, 'positive_samples')


OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'best_model_with_mask.pth')
FEATURE_SAVE_PATH = os.path.join(OUTPUT_DIR, 'extracted_features.npz')
ROC_FIGURE_PATH = os.path.join(OUTPUT_DIR, 'roc_curve.png')
UMAP_FIGURE_PATH = os.path.join(OUTPUT_DIR, 'UMAP.png')

modality_keys = {
    'seq': {
        'neg_data_key': 'OPDNASeq',
        'neg_mask_key': 'OPDNASeqmask',
        'pos_data_key': 'DNASeq',
        'pos_mask_key': 'DNASeqmask'
    },
    'snp': {
        'neg_data_key': 'OPDNASNP',
        'neg_mask_key': 'OPDNASNPmask',
        'pos_data_key': 'DNASNP',
        'pos_mask_key': 'DNASNPmask'
    },
    'variant': {
        'neg_data_key': 'OPDNASV',
        'neg_mask_key': 'OPDNASVmask',
        'pos_data_key': 'DNASV',
        'pos_mask_key': 'DNASVmask'
    },
    'methylation': {
        'neg_data_key': 'OPDNA5mc',
        'neg_mask_key': 'OPDNA5mcmask',
        'pos_data_key': 'DNA5mc',
        'pos_mask_key': 'DNA5mcmask'
    },
    'expression': {
        'neg_data_key': 'OPRNATPM',
        'neg_mask_key': 'OPRNATPMmask',
        'pos_data_key': 'RNATPM',
        'pos_mask_key': 'RNATPMmask'
    },
    'm6a': {
        'neg_data_key': 'OPRNAmod',
        'neg_mask_key': 'OPRNAmodmask',
        'pos_data_key': 'RNAmod',
        'pos_mask_key': 'RNAmodmask'
    },
    'DNA6mA': {
        'neg_data_key': 'OPDNA6mA',
        'neg_mask_key': 'OPDNA6mAmask',
        'pos_data_key': 'DNA6mA',
        'pos_mask_key': 'DNA6mAmask'
    }
}

modalities = ['seq', 'snp', 'variant', 'methylation', 'expression', 'm6a', 'DNA6mA']

xgb_modalities = ['DNA6mA', 'methylation','m6a', 'expression']
