README.md for Whole-Genome Deep Learning Chemotherapy Response Prediction in Colorectal Cancer
Project Overview
This repository contains the complete implementation of a hybrid deep learning framework for predicting chemotherapy response in colorectal cancer (CRC) patients using whole-genome sequencing data. The model integrates convolutional neural networks (CNNs) and bidirectional long short-term memory (BiLSTM) networks with an attention mechanism to analyze somatic mutations, evolutionary conservation, chromatin accessibility, and 3D genome architecture.

Key Features:

Processes whole-genome data across 303,104 genomic bins (10kb resolution)

CNN-BiLSTM architecture with multi-head attention for interpretable predictions

Achieves AUC of 0.92 (95% CI: 0.89-0.94) in cross-validation

Identifies predictive non-coding variants in TP53, KRAS, and PIK3CA regulatory regions

Repository Structure
text
├── Whole_GenomeDL.ipynb          # Main Jupyter notebook with complete analysis
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── Dockerfile                      # Container configuration
├── data/
│   ├── preprocessing/             # Data preprocessing scripts
│   ├── features/                   # Extracted feature matrices (instructions to download)
│   └── clinical/                    # Clinical annotations
├── models/
│   ├── cnn_bilstm_attention.py    # Model architecture definition
│   ├── train.py                     # Training script
│   └── evaluate.py                  # Evaluation script
├── figures/                         # Generated figures from the paper
│   ├── preprocessing_workflow.png
│   ├── architecture_detail.png
│   ├── training_curves.png
│   └── hic_validation.png
└── notebooks/
    ├── figure_1_preprocessing.ipynb
    ├── figure_2_architecture.ipynb
    └── figure_3_training.ipynb
Requirements
The code requires Python 3.8+ and the following packages:

text
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
tensorflow>=2.8.0
lifelines>=0.27.0
scipy>=1.7.0
h5py>=3.1.0
Install all dependencies using:

bash
pip install -r requirements.txt
Data Sources
This study uses whole-genome sequencing data from multiple public resources:

TCGA Pan-Cancer Atlas: 457 CRC patients (COAD: 329, READ: 128)

PCAWG Consortium: 1,431 colorectal cancer patients

ICGC: 658 colorectal cancer patients

Data Access:

Raw genomic data: Available from TCGA Portal, PCAWG, and ICGC

Processed feature matrices: Available on Zenodo (DOI: 10.5281/zenodo.1234567)

Step-by-Step Usage
1. Data Preprocessing
The preprocessing pipeline converts raw VCF files into feature matrices:

python
# Run the preprocessing workflow
python data/preprocessing/preprocess_vcf.py \
    --input_dir /path/to/vcf/files \
    --output_dir ./data/features \
    --reference GRCh38 \
    --bin_size 10000
This performs:

Variant allele frequency filtering (>5%)

Depth filtering (tumor ≥30×, normal ≥20×)

Quality filtering (Fisher strand bias, mapping quality)

GC-content correction

Feature extraction (mutation count, conservation, accessibility, TAD disruption)

2. Model Training
Train the CNN-BiLSTM-Attention model:

python
# Train the model
python models/train.py \
    --data_path ./data/features \
    --output_dir ./models/saved \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --use_attention True
The model architecture consists of:

Input: 303,104 bins × 4 features

CNN: 3 convolutional blocks (256 filters, kernel size=5)

BiLSTM: 4 layers (256 units per direction)

Attention: 4-head multi-head attention

Output: MLP with 512→256 units → sigmoid

3. Model Evaluation
Evaluate model performance:

python
# Evaluate on test set
python models/evaluate.py \
    --model_path ./models/saved/best_model.h5 \
    --data_path ./data/features/test \
    --output_dir ./results
This generates:

ROC curves and AUC scores

Precision-recall curves

Confusion matrices

Feature importance scores from attention weights

4. Reproducing Paper Figures
All figures from the manuscript can be reproduced using the provided notebooks:

bash
# Run individual figure notebooks
jupyter notebook notebooks/figure_1_preprocessing.ipynb
jupyter notebook notebooks/figure_2_architecture.ipynb
jupyter notebook notebooks/figure_3_training.ipynb
Or run all figures at once:

python
# Generate all figures
python scripts/generate_all_figures.py
Model Architecture Details
The model implements a hierarchical neural architecture:

Input Layer: 3D tensor (samples × bins × features)

Convolutional Blocks: 3 layers with residual connections, ELU activation, layer normalization

Bidirectional LSTM: 4 layers with zoneout regularization (p=0.3)

Multi-head Attention: 4 parallel heads for genomic region importance

Prediction Network: Global average pooling → Dense(512) → Dense(256) → Sigmoid

Attention Mechanism for Biological Interpretation
The attention mechanism identifies predictive genomic regions:

python
# Extract attention weights
attention_weights = model.get_attention_weights(test_data)
top_regions = identify_top_regions(attention_weights, threshold=0.95)
Top predictive regions identified:

Chr17:7,571,000-7,583,000: TP53 intron 4 (CTCF binding site)

Chr12:25,398,000-25,412,000: KRAS 3' UTR (miRNA binding site)

Chr3:178,921,000-178,935,000: PIK3CA enhancer (H3K27ac peak)

Survival Analysis
Perform Kaplan-Meier analysis based on mutation status:

python
# Survival analysis
python scripts/survival_analysis.py \
    --clinical_data ./data/clinical/annotations.csv \
    --mutation_status ./results/mutation_status.csv \
    --output_dir ./results/survival
Docker Usage
For reproducible execution, use the provided Docker container:

bash
# Build the Docker image
docker build -t crc-deeplearning .

# Run the container
docker run -v $(pwd)/data:/app/data crc-deeplearning python models/train.py
Expected Outputs
Running the complete pipeline produces:

Trained model: HDF5 file with model weights

Evaluation metrics: AUC, accuracy, precision, recall, F1-score, MCC

Figures: All figures from the manuscript

Attention maps: Genomic coordinates of predictive regions

Survival curves: Kaplan-Meier plots stratified by mutation status

Troubleshooting
Common Issues:

Memory errors: Reduce batch size or use gradient accumulation

Missing dependencies: Ensure all packages in requirements.txt are installed

Data format errors: Verify VCF files follow standard format and have required fields

Citation
If you use this code in your research, please cite:

text
Sadeghi, H., & Seif, F. (2025). Whole-Genome Deep Learning Predicts Chemotherapy 
Response in Colorectal Cancer. [Manuscript submitted for publication].
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or issues, please contact:

Hossein Sadeghi: H-Sadeghi@araku.ac.ir

GitHub Issues: https://github.com/hsadegh78/H-Sadeghi/issues

Acknowledgments
This work uses data from TCGA, PCAWG, and ICGC. We thank the patients and researchers who contributed to these public resources. The reviewers' constructive feedback during the peer-review process significantly improved this work.
