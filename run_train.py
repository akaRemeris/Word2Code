"""Main script which triggers whole taining/evaluation process."""

import argparse

import yaml
from dataclasses_utils import get_dataloader
from general_utils import init_random_seed
from preprocess_tools import Tokenizer, read_src_tgt_dataset
from rnn_transformer import TransformeRNN
from train_eval_utils import run_train_eval_pipeline, save_model


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-config',
                        dest="config",
                        default='./config.yaml',
                        type=str,
                        help='')
    args = parser.parse_args()
    with open(args.config, "r", encoding='utf-8') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    init_random_seed(config['seed'])
    train_data = read_src_tgt_dataset(path=config['dataset_path'],
                                        filename=config['train_dataset'])
    eval_data = read_src_tgt_dataset(path=config['dataset_path'],
                                        filename=config['eval_dataset'])
    tokenizer = Tokenizer()

    train_tokenized = tokenizer.tokenize_corpus(train_data)
    eval_tokenized = tokenizer.tokenize_corpus(eval_data)
    tokenizer.build_vocabulary(train_tokenized)
    train_encoded = tokenizer.encode_corpus(train_tokenized)
    eval_encoded = tokenizer.encode_corpus(eval_tokenized)
    train_dataloader = get_dataloader(train_encoded, config)
    eval_dataloader = get_dataloader(eval_encoded, config)

    vocabulary_size = len(tokenizer.vocabulary)
    init_random_seed(config['seed'])
    model = TransformeRNN(src_vocabulary_size=vocabulary_size,
                            tgt_vocabulary_size=vocabulary_size,
                            config=config)
    run_train_eval_pipeline(model=model,
                            train_dataloader=train_dataloader,
                            eval_dataloader=eval_dataloader,
                            config=config)
    if config['save_model']:
        save_model(tokenizer, model, config)
