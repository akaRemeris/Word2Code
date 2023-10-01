"""Classes and functions for data batch processing."""

from typing import Tuple

import torch
from general_utils import PAD_IDX
from torch.utils.data import DataLoader, Dataset


class Seq2SeqDataset(Dataset):
    """
    PyTorch Dataset subclass for sequence-to-sequence models.
    Takes two arguments:
        src_sequence:
            2D list or array of integers,
            where each row represents a source sequence
        tgt_sequence:
            2D list or array of integers,
            where each row represents a target sequence

    The getitem method returns a tuple containing two
    PyTorch LongTensor instances: one for the src_sequence
    at a specified index, and one for the tgt_sequence at the same index
    with the first element typically excluded as it is usually a start token.
    These values can be passed as inputs to a PyTorch DataLoader
    for training a sequence-to-sequence model.
    """

    def __init__(self, src_sequence: torch.Tensor,
                 tgt_sequence: torch.Tensor) -> None:
        """
        Constructor method for Seq2SeqDataset

        Args:
            src_sequence:
                2D list or array of integers,
                where each row represents a source sequence

            tgt_sequence:
                2D list or array of integers,
                where each row represents a target sequence.
        """

        self.src_sequence = src_sequence
        self.tgt_sequence = tgt_sequence

    def __len__(self) -> int:
        """
        Returns the number of sequences in the source sequence array.
        """

        return len(self.src_sequence)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor,
                                             torch.LongTensor]:
        """
        Gets the source sequence and target sequence for a given index.

        Args:
        - idx: index of the sequence to retrieve.

        Returns:
        A tuple containing two PyTorch LongTensor instances:
            src_sequence:
                A LongTensor representing the source
                sequence at the given index
            tgt_sequence:
                A LongTensor representing the target sequence
                at the given index with the first element
                typically excluded as it is usually a start token
        """

        src_sequence = torch.LongTensor(self.src_sequence[idx])
        tgt_sequence = torch.LongTensor(self.tgt_sequence[idx][1:])
        return (src_sequence, tgt_sequence)

class InferenceDataset(Dataset):
    def __init__(self, encoded_data) -> None:
        super().__init__()
        self.encoded_data = encoded_data
    
    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int) -> torch.LongTensor:
        src_sequence = torch.LongTensor(self.encoded_data[idx])
        tgt_sequence = torch.LongTensor(self.encoded_data[idx])
        return (src_sequence, tgt_sequence)

def custom_collate(data):
    """
    This function takes a list of data and returns
    a dictionary containing the padded source sequences,
    padded target sequences, and source sequence lengths.

    Args:
        data:
            A list of tuples containing source sequences and target sequences.

    Returns:
        A dictionary with keys src_ids, tgt_ids, and src_lengths.

    """
    # Separate the source and target sequences from the list of input tuples
    src_sequences, tgt_sequences = zip(*data)

    # Calculate the length of each source sequence in src_sequences
    src_lengths = [len(seq) for seq in src_sequences]

    # Pad the sequences to ensure they have same length
    src_sequences = torch.torch.nn.utils.rnn.pad_sequence(sequences=src_sequences,
                                                          batch_first=True,
                                                          padding_value=PAD_IDX)
    
    tgt_sequences = torch.torch.nn.utils.rnn.pad_sequence(sequences=tgt_sequences,
                                                          batch_first=True,
                                                          padding_value=PAD_IDX)

    # Return a dictionary with the padded source
    # and target sequences, as well as the source lengths
    return {'src_ids': src_sequences,
            'tgt_ids': tgt_sequences,
            'src_lengths': src_lengths}


def get_dataloader(encoded_data: [dict, list],
                   batch_size: int,
                   drop_last: bool = True) -> DataLoader:
    if isinstance(encoded_data, dict):
        dataset = Seq2SeqDataset(encoded_data['SRC'], encoded_data['TGT'])
    else:
        dataset = InferenceDataset(encoded_data)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            collate_fn=custom_collate,
                            drop_last=drop_last)
    return dataloader
