"""Main script which triggers whole taining/evaluation process."""

import argparse

import yaml
from dataclasses_utils import get_dataloader
from general_utils import init_random_seed
from preprocess_tools import Tokenizer, read_src_tgt_dataset
from rnn_transformer import TransformeRNN
from train_eval_utils import run_train_eval_pipeline, save_model

def inference(src_row: str,
              model: TransformeRNN,
              tokenizer: Tokenizer):
    model.eval()
    tokenized_sample = tokenizer.tokenize_doc(src_row)
    encoded_sample = [tokenizer.encode_doc(tokenized_sample)]
    inference_dataloader = get_dataloader(encoded_sample,
                                          batch_size=1,
                                          drop_last=False)
    for batch in inference_dataloader:
        model_output = model.forward(**batch, teacher_forcing_ratio=0.0)
        predicted_ids = model_output.argmax(2).tolist()
    decoded_doc = tokenizer.decode_doc(predicted_ids[0])
    return ' '.join(decoded_doc)

def main(config: dict):

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
    train_dataloader = get_dataloader(train_encoded, config['batch_size'])
    eval_dataloader = get_dataloader(eval_encoded, config['batch_size'])

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-config',
                        dest="config",
                        default='./config.yaml',
                        type=str,
                        help='')
    args = parser.parse_args()
    with open(args.config, "r", encoding='utf-8') as ymlfile:
        config_dict = yaml.load(ymlfile, Loader=yaml.FullLoader)
    main(config_dict)
