#Human-like-visual-attention-embedding-framework
# Guided-Search Visual Attention (GVA) Framework
## Code for "Embedding human-like visual attention for brain-aligned and high-performance visual intelligence"

### Overview
This repository provides the official implementation of the human-like visual attention embedding framework from our manuscript. It includes the core GVA module for target detection models, and full code for brain-model neural alignment analysis via Representational Similarity Analysis (RSA).

### File Descriptions
- `Guided-Search Visual Attention Module.py`: Core implementation of the dual-pathway GVA module (bottom-up stimulus-driven saliency + top-down target-directed guidance)
- `EEG_RSM.py`: Code to build neural Representational Similarity Matrices (RSMs) from preprocessed human EEG data
- `Model_RSM.py`: Code to extract hierarchical features from detection models and generate model RSMs
- `RSA.py`: Core script to run RSA and quantify brain-model neural representational alignment
- `Lab_RF_10_5.pkl`: Pre-trained random forest classifier for the GVA module's top-down guidance pathway
- `LICENSE`: Open-source license for this repository

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NumPy, SciPy, scikit-learn
- OpenCV-Python

### Basic Usage
1.  **Integrate GVA Module**: Import the GVA module to embed human-like attention into mainstream target detection architectures (Faster R-CNN, YOLO series, RetinaNet, CenterNet).
2.  **Run RSA Analysis**: Generate neural RSMs with `EEG_RSM.py`, build model RSMs with `Model_RSM.py`, then run `RSA.py` to calculate alignment metrics.

### Citation
If you use this code in your research, please cite our manuscript:
