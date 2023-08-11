# AMDRD (Analysis Model of Discourse Relations within a Document)
This repository contains the source code and necessary instructions for the AMDRD model, which focuses on the extraction of discourse relations within a document.

# Folder Structure
src/: Contains the main source code.

boundary_extraction/: Contains code for ADU tagging and boundary extraction.
relation_identification/: Contains code for ADU relation identification.
data/: Holds publicly available datasets from other research teams used in this study.

LREC_dataset/: Dataset used for training the argument structure extraction model.
## Requirements
Make sure you have the following dependencies installed:

pytorch
transformers
numpy
sklearn
nltk
tqdm
# Code Functionality
### boundary_extraction/
Functionality: Trains an ADU boundary recognition model on the CMV annotated dataset.

Required Files: Depends on the LREC_dataset/.

Parameters: CMV dataset path ($LREC_PATH), storage location ($SAVE_PATH).

Training: Execute: python train.py $LREC_PATH $SAVE_PATH

Testing: Execute: python test.py $LREC_PATH $SAVE_PATH

### relation_identification/
Functionality: Trains a relation identification model on the CMV annotated dataset.

Required Files: Depends on the LREC_dataset/.

Parameters: CMV dataset path ($LREC_PATH), pretrained model path ($PRETRAIN_PATH), storage location ($SAVE_PATH).

Data Preprocessing: Execute: python prepare_relation_data.py $LREC_PATH $SAVE_PATH

BERT Training: Execute: python train.py $PRETRAIN_PATH $SAVE_PATH

XGBoost Training: Execute: python ensemble.py $PRETRAIN_PATH $SAVE_PATH

Testing: Execute: python test.py $PRETRAIN_PATH $SAVE_PATH

Feel free to explore the code within the respective folders for more detailed functionalities.

# Obtaining LREC_dataset
For the LREC_dataset, please visit the author's website to download the dataset:
http://katfuji.lab.tuat.ac.jp/nlp_datasets/

# How to Cite the Model
If you use this model in your research or work, please consider citing the following study that this submodule is a part of:

Fa-Hsuan Hsiao, An-Zi Yen, Hen-Hsen Huang, and Hsin-Hsi Chen (2022). "Modeling Inter Round Attack of Online Debaters for Winner Prediction." In Proceedings of the Web Conference 2022, April 25-29, online, hosted by Lyon, France. (acceptance rate=17.7%, 323 of 1822 submissions).


