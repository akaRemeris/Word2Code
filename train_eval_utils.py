"""Utilities for training, evaluating, saving model and logging run results."""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import evaluate
import torch
from preprocess_tools import Tokenizer
from general_utils import EOS_IDX, PAD_IDX
from rnn_transformer import TransformeRNN
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


@dataclass
class ModelLog(object):
    """
    Class serves logging purposes of train and eval loss and any chosen metric.

    Args:
        train_loss_log (list):
            List contains train loss values for each iteration/epoch.
        eval_loss_log (list):
            List contains evaluation loss values for each iteration/epoch.
        train_metric_log (list):
            List contains train metric values for each iteration/epoch.
        eval_metric_log (list):
            List contains evaluation metric values for each iteration/epoch.
    """
    train_loss_log: List[float] = field(default_factory=list)
    eval_loss_log: List[float] = field(default_factory=list)
    train_metric_log: List[float] = field(default_factory=list)
    eval_metric_log: List[float] = field(default_factory=list)


def truncate_sequence(sequence: List) -> List:
    """
    Truncates a given sequence of tokens up to the end-of-sequence token
    (EOS_TOKEN) if it exists, and returns the truncated sequence.

    Args:
    - sequence (List):
        A list of tokens representing the sequence.

    Returns:
    - List:
        A truncated sequence up to the index of the end-of-sequence
        token (if present), or the original sequence if it does not
        contain the end-of-sequence token.

    Example usage:
    ```
    trunc_seq = truncate_sequence(sequence=[4, 3, 2, 1, 0])
    # The output would be: [4, 3]
    ```
    """
    
    # trim sequence if it contains EOS_IDX, else return unchanged
    try:
        eos_sequence_idx = sequence.index(EOS_IDX)
        return sequence[:eos_sequence_idx]
    except ValueError:
        return sequence


def convert_to_str(sequence: List[List[int]]) -> List[str]:
    # Convert each element of each list to str and join into row
    return ' '.join(list(map(str, sequence)))


def compute_rouge_metric(preds: List[List[int]],
                         refs: List[List[int]]) -> float:
    """
    Computes the ROUGE L score between two lists of integer sequences.

    Args:
        preds (List[List[int]]): A list of integer
            sequences representing predicted summaries.
        refs (List[List[int]]): A list of integer
            sequences representing reference summaries.

    Returns:
        float: A floating point number representing the ROUGE L score.
    """
    # Torch's tolist() tends to squeeze original tensor with single value,
    # so we have to restore dimentiality
    if isinstance(preds[0], int):
        preds = [preds]
        refs = [refs]

    # trancuate sequence to exclude padding ids
    preds = list(map(truncate_sequence, preds))
    refs = list(map(truncate_sequence, refs))

    # convert list of ints into rows
    preds = list(map(convert_to_str, preds))
    refs = list(map(convert_to_str, refs))

    # compute the ROUGE L score
    # ain't a good thing
    rouge_score = evaluate.load('rouge')
    score = rouge_score.compute(references=refs, predictions=preds)
    return score['rougeL']


def compute_em_metric(preds, refs):
    """
        Computes the EM score between two lists of integer sequences.

    Args:
        preds (List[List[int]]): A list of integer predicted sequences.
        refs (List[List[int]]): A list of integer reference sequences.

    Returns:
        float: A floating point number representing the EM score.
    """

    # Torch's tolist() tends to squeeze original tensor with single value
    # so we have to restore dimentiality
    if isinstance(preds[0], int):
        preds = [preds]
        refs = [refs]

    # trancuate sequence to exclude padding ids
    preds = list(map(truncate_sequence, preds))
    refs = list(map(truncate_sequence, refs))

    # count all position matches
    matches_counter = 0
    for pair in zip(preds, refs):
        matches_counter += int(pair[0] == pair[1])

    # normalize by length
    exact_match = matches_counter / len(preds)
    return exact_match


def produce_epoch_train(model: TransformeRNN, optimizer: torch.optim.AdamW,
                        loss_foo: Callable, train_dataloader: DataLoader,
                        metric_foo: Callable = None) -> Tuple[float,
                                                              float,
                                                              float]:
    """
    Trains a PyTorch model using the provided optimizer
    and loss function on the given training data.

    Args:
        model (TransformeRNN):
            PyTorch model to train
        optimizer (AdamW):
            Optimizer for updating the model's parameters
        loss_foo (Callable):
            Loss function to compute the training loss
        train_dataloader (DataLoader):
            Training data loader containing training examples
        metric_foo (Callable):
            Function to compute additional metrics during training

    Returns:
        epoch_mean_loss (float): mean loss for the epoch
        training_time (float): time taken to train the epoch
        metric (float): additional metric value computed during training
    """

    # Start time for training
    train_start_time = time.time()

    # Set the model to training mode
    model.train()

    # Initialize variables for batch loss
    # epoch predictions, and epoch reference targets
    batch_loss = 0
    epoch_preds = []
    epoch_refs = []

    # Iterate through the training data loader
    for batch_idx, batch in enumerate(train_dataloader):
        # Zero out the gradients
        optimizer.zero_grad()

        # Get the model output for the current batch
        model_output = model(**batch)

        # Append the current batch predictions
        # and references to the overall epoch lists
        epoch_preds += model_output.argmax(2).tolist()
        epoch_refs += batch['tgt_ids'].tolist()

        # Compute the loss and do backpropagation
        target_ids = batch['tgt_ids'].flatten()
        predicted_ids = model_output.view(-1, model.tgt_vocabulary_size)
        loss = loss_foo(predicted_ids, target_ids)
        loss.backward()
        optimizer.step()

        # Add the current batch loss to the overall epoch loss
        batch_loss += loss.item()

    # Compute any additional metric value if provided
    metric = (
        metric_foo(epoch_preds, epoch_refs)
        if metric_foo is not None
        else None
    )

    # Compute the mean loss for the epoch and training time
    epoch_mean_loss = batch_loss / (batch_idx + 1)
    training_time = time.time() - train_start_time

    # Return the epoch mean loss, training time, and optional metric value
    return epoch_mean_loss, training_time, metric


def produce_epoch_eval(model: TransformeRNN,
                       loss_foo: Callable,
                       eval_dataloader: DataLoader,
                       metric_foo: Callable = None) -> Tuple[float,
                                                             float,
                                                             float]:
    """
    Evaluates a model on an evaluation dataset
    using the provided loss and optional metric function.

    Args:
        model (torch.nn.Module):
            The model to evaluate.
        loss_foo (Callable):
            Loss function to compute the output loss on each batch.
        eval_dataloader (torch.utils.data.DataLoader):
            The evaluation data loader.
        metric_foo (Callable):
            The optional metric function to compute a performance metric.

    Returns:
        tuple: A tuple of (epoch_mean_loss, procedure_time, metric) where:
            epoch_mean_loss (float):
                The mean batch loss over the evaluation dataset.
            procedure_time (float):
                The time taken to complete the evaluation.
            metric (float or None):
                The computed metric if metric_foo was provided, otherwise None.
    """

    start_time = time.time()
    model.eval()
    batch_loss = 0
    epoch_preds = []
    epoch_refs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            # get the model output
            model_output = model(**batch, teacher_forcing_ratio=0.0)

            # add the model output
            # and target to epoch predictions and references list
            epoch_preds += model_output.argmax(2).tolist()
            epoch_refs += batch['tgt_ids'].tolist()

            # compute the loss on the batch
            target_ids = batch['tgt_ids'].flatten()
            predicted_ids = model_output.view(-1, model.tgt_vocabulary_size)
            loss = loss_foo(predicted_ids, target_ids)
            batch_loss += loss.item()

    # compute the metric if it was provided
    metric = (
        metric_foo(epoch_preds, epoch_refs)
        if metric_foo is not None
        else None
    )

    # compute the epoch mean loss and procedure time
    epoch_mean_loss = batch_loss / (batch_idx + 1)
    procedure_time = time.time() - start_time

    return epoch_mean_loss, procedure_time, metric


def epochwise_train_eval_with_logging(model: TransformeRNN,
                                      train_dataloader: DataLoader,
                                      eval_dataloader: DataLoader,
                                      optimizer: torch.optim,
                                      loss_foo: Callable,
                                      max_epoch: int,
                                      metric_foo: Callable,
                                      verbose: bool = True) -> ModelLog:
    """
    General train/eval procedure with epoch strategy
    and basic logging functionality.

    Args:
        model (TransformeRNN):
            Model, subclass of torch.nn.Module to be trained.
        train_dataloader (DataLoader):
            Torch data iterator which returns
            shuffled batches of training data.
        eval_dataloader (DataLoader):
            Torch data iterator which
            returns shuffled batches of training data.
        optimizer (torch.optim):
            Torch initialized optimizer.
        loss_foo (Callable):
            Function for loss computation that model aims to optimize.
        max_epoch (int):
            Number of full data iterations.
        metric_foo (Callable):
            Function which computes main metric of model prediction.
        verbose (bool, optional):
            Defines whether or not print in cmd training stats.
            Defaults to True.

    Returns:
        logger (ModelLog):
            Object which contains lists representing
            loss and other metrics logs.
    """
    # initialize logger for current run
    logger = ModelLog()

    # Loop through epochs
    for epoch_idx in range(max_epoch):

        # Train on data and record training loss and metric
        train_loss, train_time, train_metric = produce_epoch_train(
            model=model,
            optimizer=optimizer,
            loss_foo=loss_foo,
            train_dataloader=train_dataloader,
            metric_foo=metric_foo
        )
        logger.train_loss_log.append(train_loss)
        logger.train_metric_log.append(train_metric)

        # Evaluate on data and record eval loss and metric
        eval_loss, eval_time, eval_metric = produce_epoch_eval(
            model=model,
            loss_foo=loss_foo,
            eval_dataloader=eval_dataloader,
            metric_foo=metric_foo
        )

        logger.eval_loss_log.append(eval_loss)
        logger.eval_metric_log.append(eval_metric)

        # Print some statistics about the current epoch if requested
        if verbose:
            print((
                "Epoch №: {}"
                "\t| Training time: {}"
                "\t| Evaluation time: {}"
                "\t| train loss: {:5.3f}"
                "\t| eval loss: {:5.3f}"
                "\t| train metric: {:5.3f}"
                "\t| eval metric: {:5.3f} ")
                .format(epoch_idx + 1,
                        time.strftime('%M:%S', time.gmtime(train_time)),
                        time.strftime('%M:%S', time.gmtime(eval_time)),
                        train_loss,
                        eval_loss,
                        train_metric,
                        eval_metric
                        )
            )

    return logger


def commit_to_tensorboard(config: dict, logger: ModelLog) -> None:
    """
    Save data logged in ModelLog object in files
    for Tensorboard to read and display.

    Args:
        config (dict):
            General config which contains hyperparameters of the model,
            where the name of the parameter is config's key
            and the value is config's value.
        logger (ModelLog):
            Object which contains lists representing loss and metric logs.
    """

    # Create SummaryWriter if logs are requested, otherwise set to None
    model_log_path = config['model_log_path']
    model_log_name = config['log_run_name']
    writer = SummaryWriter(model_log_path + model_log_name)
    # If logs are requested, add them to tensorboard via SummaryWriter

    # get number of train/eval records
    assert_message = ("Number of evaluation records"
                      "have to be the same as training ones.")
    train_operations_number = len(logger.train_loss_log)
    eval_operations_number = len(logger.eval_loss_log)
    assert train_operations_number == eval_operations_number, assert_message

    for iteration_idx in range(train_operations_number):

        # get loss and metric for the current iteration_idx
        current_train_loss = logger.train_loss_log[iteration_idx]
        current_train_metric = logger.train_metric_log[iteration_idx]
        current_eval_loss = logger.eval_loss_log[iteration_idx]
        current_eval_metric = logger.eval_metric_log[iteration_idx]

        # create 4 separate plots for each train/eval * loss/metric logs
        writer.add_scalar('Loss/train', current_train_loss, iteration_idx)
        writer.add_scalar('EM/train', current_train_metric, iteration_idx)
        writer.add_scalar('Loss/eval', current_eval_loss, iteration_idx)
        writer.add_scalar('EM/eval', current_eval_metric, iteration_idx)

        # create 2 plots with combined train and eval logs for comparison
        writer.add_scalars('Loss/combined', {
            'train': current_train_loss,
            'eval': current_eval_loss
        }, iteration_idx)
        writer.add_scalars('EM/combined', {
            'train': current_train_metric,
            'eval': current_eval_metric
        }, iteration_idx)

    writer.close()


def run_train_eval_pipeline(model: TransformeRNN,
                            train_dataloader: DataLoader,
                            eval_dataloader: DataLoader,
                            config: dict) -> None:
    """
    General function for managing model's training and evaluation,
    it initilizes training utilities according to hyperparameters
    set in config and allows to commit results to tensorboard if requested.

    Args:
        model (TransformeRNN):
            Seq2Seq autoregression model with BiLSTM as encoder's
            backbone and LSTM as decoder's backbone.
        train_dataloader (DataLoader):
            Torch's DataLoader which returns padded SRC and TGT ids
            of evaluation dataset for model input.
        eval_dataloader (DataLoader):
            Torch's DataLoader which returns padded SRC and TGT ids
            of evaluation dataset for model input.
        config (dict):
            General config which contains hyperparameters of the model,
            where the name of the parameter is config's key and
            the value is config's value
    """
    
    loss_foo = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), config['learning_rate'])
    logger = epochwise_train_eval_with_logging(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        loss_foo=loss_foo,
        max_epoch=config['max_epoch'],
        metric_foo=compute_em_metric,
        verbose=config['verbose']
    )

    if config['save_logs']:
        commit_to_tensorboard(config, logger)


def save_model(tokenizer: Tokenizer, model: TransformeRNN, config: dict) -> None:
    """
    Function saves model in .pth file with 3 separate keys:
        model for model object,
        model_state_dict for model weights,
        vocabulary for pre-made vocabulary

    Args:
        vocabulary (dict):
            Dictionary with tokens as keys and token ids
            as values on which model was trained.
        model (TransformeRNN):
            Object of TransformeRNN, subclass of torch.nn.Module
        config (dict):
            General config which contains hyperparameters of the model,
            where the name of the parameter is config's key
            and the value is config's value.
    """

    checkpoint = {
        "model": model,
        "model_state_dict": model.state_dict(),
        "tokenizer": tokenizer
    }

    model_file_name = ''.join([config['model_path'],
                               config['log_run_name'],
                               str(config['seed']),
                               '.pth'])
    torch.save(checkpoint, model_file_name)
