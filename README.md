# torch_LSTM2LSTM

## General description
Unified implementation of a neural network based on the Transformer architecture. The cores of the Transformer's modules are recurrent neural networks, Encoder's backbone is - BiLSTM, and Decoder's backbone is - LSTM. The connection between modules is established through the mechanism of additive attention.
Original model archetecture including attention mechanism were introdused in [2014 paper](https://arxiv.org/abs/1409.0473), current model implemented based of a [demonstrational Pytorch implementation](https://github.com/bentrevett/pytorch-seq2seq).

### Input data format description
<p>
Model is pretty much ready to deal with any kind of simple seq2seq tasks (MT, abstract summarization, so on). You just have to provide your own task specific data and define tokenization functions. After training, you may find useful a generation pipeline from generation.py.
</p>


## Project structure

* **run_train.py** - main script for model training and validation.
* **rnn_transformer.py** - module contains seq2seq model definition with all it's sub-modules e.i. encoder, decoder, attention.
* **preprocess_utils.py** - functions and classes for data preprocess e.i. Tokenizer for tokenization,
vocabulary building, encoding, e.t.c.
* **dataclasses_utils.py** - classes for iteratable batching of preprocessed data.
* **train_eval_utils.py** - functions for model training and validation.
* **general_utils.py** - model special tokens, ids, and random seed initialization function definition
* **generation.py** - contains beamsearch and high lvl inference functions (str2str)
* **task_specific_utilities.py** - contain definitions of custom tokenizers.

* `/model/` - default directory for model's files.
* `/experiments/` - default directory with TensorBorad logs.

* **config.yaml** - general config with following parameters:
    * paths and file names:
        * datasets_path - path to files with saved dataset.
        * model_path - path where model is to be saved.
        * train_dataset - dictionary with two filenames which contain source and target sequences for training
        * train_dataset - dictionary with two filenames which contain source and target sequences for validation
    * logging and saving options:
        * log_run_name - name of pth file and directory contains tensorboard's logs
        * save_logs - flag if logs (loss and metric) are to be saved.
        * save_model - flag if model is to be saved after tranining.
        * verbose - flag if training and validation information is to be printed in the console.

    * data processing parameters:
        * custom_tokenization - flag if to use custom defiened tokenizers from task_specific_utilities.py
        * src_min_token_count - minimum documents containing token for it to be included into vocabulary of source sequences
        * src_max_token_freq - maximum documents containing token/len(corpus) for it to be included into vocabulary of source sequences
        * tgt_min_token_count - minimum documents containing token for it to be included into vocabulary of target sequences
        * tgt_max_token_freq - maximum documents containing token/len(corpus) for it to be included into vocabulary of target sequences

    * model parameters:
        * embedding_size - size of encoder and decoder embeddings
        * hidden_size - size of encoder's and decoder's LSTMs hidden states
        * dropout - embedding dropout probability

    * train/eval parameters:
        * learning_rate - gradient step scale
        * max_epoch - number of full data iterations
        * batch_size - number of chunks in dataset iterators
        * seed - general initial number of PRNG
        * metric - wip, untill then set to None
        * device - 'cuda' or 'cpu'
        

## How to run
Config by default config.yaml:
```
python run_train.py
```
Custom config:
```
python run_decoder.py -config <NAME_OF_CONFIG>
```

## References
* https://arxiv.org/abs/1409.0473
* https://github.com/bentrevett/pytorch-seq2seq