# DeepSentiPeer
Harnessing sentiment in the review texts to recommend peer-review decisions

```@inproceedings{ghosal-etal-2019-deepsentipeer,
    title = "{D}eep{S}enti{P}eer: Harnessing Sentiment in Review Texts to Recommend Peer Review Decisions",
    author = "Ghosal, Tirthankar  and
      Verma, Rajeev  and
      Ekbal, Asif  and
      Bhattacharyya, Pushpak",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1106",
    doi = "10.18653/v1/P19-1106",
    pages = "1120--1130",
}

This repository contains the updated code to the above paper. The code now supports SciBERT sentence encoder. We currently are in the process of making further updates. Currently, only the `Recommendation Task` is supported.

The organization of is as follows:
```
.
+--`prepare_data.py`: generate and write the review and paper embeddings in .json file.
+--`utils.py`: Relevant utility code for reading and embedding data
+--`run_model.py`: Train the model (currently, supports only the Recommendation Task)
```

To get started, follow:
```
$python prepare_data.py ./2018
$python run_model.py
```

Note: `prepare_data.py` may run for many hours depending on the size of the data.

The `run_model.py` has the following settings:

```
usage: run_model.py [-h] [--batch_size BATCH_SIZE] [--dropout DROPOUT]
                    [--l2 L2] [--learning_rate LEARNING_RATE] [--mode MODE]
                    [--datadir DATADIR] [--ckpdir CKPDIR]
                    [--exp_name EXP_NAME]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size to train the model (default: 32)
  --dropout DROPOUT     dropout probability (default: 0.5)
  --l2 L2               l2 weight decay penalty (default: 0.007)
  --learning_rate LEARNING_RATE
                        learning rate for the gradient based Algorithm
                        (default: 0.001)
  --mode MODE           Task mode, choose from [RECOMMENDATION, DECISION]
                        (default: RECOMMENDATION)
  --datadir DATADIR     Path to the Dataset (default: ./2018)
  --ckpdir CKPDIR       Path to save the trained models (default: ./MODELS)
  --exp_name EXP_NAME   Name of the experiment, model and params will be saved
                        with this name (default: default)
```