### Generative QA Models For Question Answering

#### Introduction

This is the codebase for our CMSC470 project on evaluaing closed-book generative QA models for the task of question answering. 

Team members: Arjun Akkiraju, Gurmehar Cheema, Elliot Huang, Naman Molri, and Chenglei Si.


#### Progress (Last updated: 19 Nov, 2021)

Things that we have done:

- An improved version of answer evaluation with the use of RapidFuzz. This accounts for spelling variations of the answer prediction and allows for a more robust evaluation. The evaluation script is in `evaluate_fuzzy.py`. A baseline evaluation script for comparison is in `GPT3/evaluate.py`.
This part is lead by Naman. 

- A GPT-3 based QA system for solving the Quizbowl challenge. The GPT-3 model is a decoder-only autoregressive language model and performs few-shot in-context learning for Quizbowl. The code and details can be found in `GPT3`. 
This part is lead by Chenglei.

- A T5 based QA system called MACAW, which was trained on a collection of different QA datasets and recently released by AI2. We adapted it to the task of Quizbowl. The training and inference script is in `MACAW`.
This part is lead by Gurmehar. 

#### Next Steps:

- Further improving the evaluation script to better handle the corner cases. 

- Improve the finetuning of MACAW. 

- Expand our evaluation to more other types of QA datasets. 


#### Quizbowl Data Being Used

You should download the QANTA 2018 dataset from https://sites.google.com/view/qanta/resources

In this repo, `train_10.json` and `test_100.json` are sampled examples from the training and test sets. Inside `model_predictions/`, `predictions.json`  `predictions_2sents.json`  `predictions_last_sent.json` are predictions made by three different models (to the questions in `test_100.json`).

For evaluation, run `python evaluate_fuzzy.py`


#### Specific Update Messages From Team Memebers 

Naman: <br /><br />
`evaluate.py` - was updated with fuzzy matching with a fuzzy-ratio classifier for strings. This allowed us to increase our accuracy of matching predictions to correct answers (even if they were slightly off) to a much greater degree (from 63 to 88 in a set of 100 predictions and answers). Also added fucntionality for stripping answers to the best possible from an array of predictions or answers from the test set. The fuzzy.ratio score value was determined after carefully studying the answer set, understanding some possible answer formats and then tailoring the evaluate accordingly. We still haven't achieved matching to 100% degree for the test set and will consider employing methods like Word2Vec and other Deep Entity matching (for wikipedia) to increase accuracy over other datasets. 


