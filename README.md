### AMDRD (Analysis Model of Discourse Relations within a Document)
This repository contains the source code and necessary instructions for the AMDRD model, which focuses on the extraction of discourse relations within a document.

### Folder Structure
src/: Contains the main source code.

boundary_extraction/: Contains code for ADU tagging and boundary extraction.
relation_identification/: Contains code for ADU relation identification.
data/: Holds publicly available datasets from other research teams used in this study.

LREC_dataset/: Dataset used for training the argument structure extraction model.
### Requirements
Make sure you have the following dependencies installed:

pytorch
transformers
numpy
sklearn
nltk
tqdm
### Code Functionality
boundary_extraction/
Functionality: Trains an ADU boundary recognition model on the CMV annotated dataset.
Required Files: Depends on the LREC_dataset/.
Parameters: CMV dataset path ($LREC_PATH), storage location ($SAVE_PATH).
Training: Execute: python train.py $LREC_PATH $SAVE_PATH
Testing: Execute: python test.py $LREC_PATH $SAVE_PATH
relation_identification/
Functionality: Trains a relation identification model on the CMV annotated dataset.
Required Files: Depends on the LREC_dataset/.
Parameters: CMV dataset path ($LREC_PATH), pretrained model path ($PRETRAIN_PATH), storage location ($SAVE_PATH).
Data Preprocessing: Execute: python prepare_relation_data.py $LREC_PATH $SAVE_PATH
BERT Training: Execute: python train.py $PRETRAIN_PATH $SAVE_PATH
XGBoost Training: Execute: python ensemble.py $PRETRAIN_PATH $SAVE_PATH
Testing: Execute: python test.py $PRETRAIN_PATH $SAVE_PATH
Feel free to explore the code within the respective folders for more detailed functionalities.

###

This project aims to provide an effective model for analyzing discourse relations within a document. If you have any questions or suggestions, please don't hesitate to contact us. We appreciate your interest in our work!
