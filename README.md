### The Quizbowl Challenge: Models and Evaluation

You should download the QANTA 2018 dataset from https://sites.google.com/view/qanta/resources

In this repo, `train_10.json` and `test_100.json` are sampled examples from the training and test sets. `predictions.json`  `predictions_2sents.json`  `predictions_last_sent.json` are predictions made by three different models (to the questions in `test_100.json`).

For evaluation, run `python evaluate.py`

**Updates:** <br /><br />
`evaluate.py` - was updated with fuzzy matching with a fuzzy-ratio classifier for strings. This allowed us to increase our accuracy of matching predictions to correct answers (even if they were slightly off) to a much greater degree (from 63 to 88 in a set of 100 predictions and answers). Also added fucntionality for stripping answers to the best possible from an array of predictions or answers from the test set. The fuzzy.ratio score value was determined after carefully studying the answer set, understanding some possible answer formats and then tailoring the evaluate accordingly. We still haven't achieved matching to 100% degree for the test set and will consider employing methods like Word2Vec and other Deep Entity matching (for wikipedia) to increase accuracy over other datasets. 
