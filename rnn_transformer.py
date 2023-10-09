"""Seq2seq transformer model and it's modules classes."""

import heapq
import random
from typing import List, Tuple

import numpy as np
import torch
from general_utils import BOS_IDX, PAD_IDX


class Encoder(torch.nn.Module):

    """
    This class defines an encoder module that takes a sequence
    of input tokens and returns the hidden states and the last
    hidden state after passing through a bidirectional LSTM layer.
    
    Args:
        src_vocabulary_size (int):
            The size of the vocabulary of the source language
        embedding_size (int):
            The size of the embedding layer
        hidden_size (int):
            The number of units in the LSTM layer
        dropout (float):
            The probability of dropping out units during training (default=0.1)
        n_layers (int):
            The number of layers in the LSTM layer (default=1).
        pretrained_embeddings (torch.Tensor):
            Pre-trained embeddings to be used in
            the embedding layer (default=None)
    """
    
    def __init__(self, src_vocabulary_size: int,
                 embedding_size: int,
                 hidden_size: int,
                 dropout: float = 0.1,
                 n_layers: int = 1,
                 pretrained_embeddings: torch.Tensor = None) -> None:

        super(Encoder, self).__init__()
        self.get_embeddings = torch.nn.Embedding(src_vocabulary_size,
                                                 embedding_size)
        self.dropout = torch.nn.Dropout(dropout)
        
        if pretrained_embeddings is not None:
            self.get_embeddings.weight = pretrained_embeddings

        self.lstm = torch.nn.LSTM(input_size=embedding_size,
                                  hidden_size=hidden_size,
                                  num_layers=n_layers,
                                  bidirectional=True,
                                  batch_first=True)

        self.h_last_transform = torch.nn.Linear(2 * hidden_size, hidden_size)
        self.h_last_activation = torch.nn.Tanh()

    def forward(self, src_ids: torch.Tensor,
                src_lengths: torch.Tensor) -> Tuple[torch.Tensor,
                                                    torch.Tensor]:
        
        """
        Compute a forward pass through the encoder module,
        calculating encoder context representation and
        hidden state for decoder initialization.

        Args:
            src_ids (torch.tensor):
                Tensor of shape [src_len, batch_size]
                containing source sequence indices.
            src_lengths (torch.tensor):
                Tensor of shape [batch_size] containing
                the length of each source sequence in the batch.
        
        Returns:
        Tuple[torch.tensor, torch.tensor]: a tuple containing:
            h_matrices (torch.tensor):
                Tensor of shape [batch_size, src_len, 2*hidden_size]
                containing hidden states for each
                time step of each sequence in the batch
            h_last_tunned (torch.tensor):
                Tensor of shape [1, batch_size, hidden_size]
                containing the last hidden state
                of the encoder, after applying a linear
                transformation and tanh activation
        """
        # [srс_len, batch size, emb dim]
        embedded = self.dropout(self.get_embeddings(src_ids))
        
        packed_embedded_src = torch.nn.utils.rnn.pack_padded_sequence(
            input=embedded,
            lengths=src_lengths,
            enforce_sorted=False,
            batch_first=True
        )
            
        packed_h_matrices, last_hidden_cell_states = self.lstm(
            packed_embedded_src
        )

        h_matrices, _ = torch.nn.utils.rnn.pad_packed_sequence(
            sequence=packed_h_matrices,
            batch_first=True
        )

        h_last = torch.cat((last_hidden_cell_states[0][0],
                            last_hidden_cell_states[0][1]), dim=-1)

        h_last_tunned = self.h_last_activation(
            self.h_last_transform(h_last)
        ).unsqueeze(0)
        
        return h_matrices, h_last_tunned


class Attention(torch.nn.Module):

    def __init__(self, enc_h_size: int, dec_h_size: int) -> None:
        """
        Takes encoder hidden size and decoder hidden size as arguments
        and initializes the parameters of the Attention module.

        Args:
            enc_h_size (int): 
                The size of the encoder hidden state.
            dec_h_size (int): 
                The size of the decoder hidden state.
        """
        super(Attention, self).__init__()

        # Initialize linear transformation for 
        # the concatenation of encoder and decoder hidden states.
        self.h_matrix_transform = torch.nn.Linear(2 * enc_h_size + dec_h_size, 
                                                  dec_h_size)
        self.attn_activation = torch.nn.Tanh()
        
        self.v_transform = torch.nn.Linear(dec_h_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.v_transform.weight.data)

        self.out_activation = torch.nn.Softmax(-1)

    def forward(self, last_h: torch.tensor, 
                encoder_outputs: torch.tensor, 
                padding_mask: torch.tensor = None) -> torch.tensor:
        """
        Takes the last hidden state of the decoder (last_h), 
        the encoder outputs (encoder_outputs), 
        and an optional padding mask (padding_mask) as input.
        
        Returns the tensor of attention weights 
        for each position in the encoder_outputs tensor.

        Args:
            last_h (torch.tensor): 
                The last hidden state of the 
                decoder (shape: (batch_size, dec_h_size))
            encoder_outputs (torch.tensor): 
                The encoder outputs tensor 
                (shape: (batch_size, max_src_seq_len, enc_h_size * 2))
            padding_mask (torch.tensor, optional): 
                The padding mask tensor (shape: (batch_size, max_src_seq_len))

        Returns:
            attention_weights (torch.tensor): 
            The attention weights tensor (shape: (batch_size, max_src_seq_len))
        """

        # Repeat last hidden state max_src_seq_len times 
        # and concatenate with encoder_outputs
        max_seq_len = encoder_outputs.shape[1]
        last_h_repeated = last_h.repeat(max_seq_len, 1, 1).permute(1, 0, 2)
        h_vector_with_matrix = torch.cat((last_h_repeated, 
                                          encoder_outputs), dim=-1)

        # Apply linear transformation and activation function.
        h_matrix_energy = self.attn_activation(
            self.h_matrix_transform(h_vector_with_matrix)
        )

        # Apply another linear transformation and squeeze 2nd dimension.
        energy = self.v_transform(h_matrix_energy).squeeze(2)

        # If padding_mask is not None,
        # mask the energy values for padding positions.
        if padding_mask is not None:
            energy = energy.masked_fill(padding_mask, -1e10)

        # Apply Softmax function and return the attention weights.
        attention_weights = self.out_activation(energy)        
        return attention_weights

    
class Decoder(torch.nn.Module):

    """
    A decoder module for sequence-to-sequence 
    neural machine translation models.

    Args:
        tgt_vocabulary_size (int): 
            Size of the target vocabulary
        encoder_embedding_size (int): 
            Size of the encoder embedding
        decoder_embedding_size (int): 
            Size of the decoder embedding
        hidden_state_size (int): 
            Size of the decoder's hidden state
        n_layers (int): 
            Number of layers in the decoder's LSTM, defaults to 1
        dropout (float): 
            Dropout rate, defaults to 0.1
    """
    
    def __init__(
            self,
            tgt_vocabulary_size: int,
            embedding_size: int,
            hidden_state_size: int,
            n_layers: int = 1,
            dropout: float = 0.1
    ) -> None:
        
        super(Decoder, self).__init__()
        self.get_embeddings = torch.nn.Embedding(tgt_vocabulary_size,
                                                 embedding_size)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        self.attention = Attention(hidden_state_size,
                                   hidden_state_size)
        
        lstm_input_size = hidden_state_size * 2 + embedding_size
        self.lstm = torch.nn.LSTM(
            lstm_input_size,
            hidden_state_size,
            n_layers,
            batch_first=True
        )
        
        self.out_transform = torch.nn.Linear(lstm_input_size + hidden_state_size,
                                             tgt_vocabulary_size)

    def forward(self, 
                tgt_ids: torch.tensor,
                last_hidden_cell_states: torch.tensor,
                encoder_outputs: torch.tensor,
                padding_src_mask: torch.tensor = None
                ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute a forward pass through the decoder module, 
        updating the hidden and cell states and producing 
        an output representation for the current time step.

        Args:
            tgt_ids (torch.tensor): 
                Tensor of shape (batch_size, 1) representing the target
                token ids for the current time step
            last_hidden_cell_states (torch.tensor): 
                Tuple of tensors of representing
                the decoder's previous hidden and cell states
            encoder_outputs (torch.tensor): 
                Tensor of shape representing encoded 
                outputs of the source sequence
            padding_src_mask (torch.tensor): 
                Optional tensor of representing 
                the padding positions in the source sequence

        Returns:
            out_proj (torch.tensor): 
                Tensor representing the probability distribution over 
                the target vocabulary for the current time step
            last_hidden_cell_states (torch.tensor): 
                Tuple of tensors representing the decoder's updated
                hidden and cell states for the next time step.
        """

        # Get embeddings from target ids and apply dropout
        embeddings = self.dropout(self.get_embeddings(tgt_ids))

        # Compute the attention weights and attend to the encoder outputs
        attn_weights = self.attention(last_hidden_cell_states[0],
                                      encoder_outputs, 
                                      padding_src_mask
                                      ).unsqueeze(1)
        context = attn_weights.bmm(encoder_outputs)  # (N, 1, H)
        
        # Concatenate embeddings and attention context and feed to the LSTM
        rnn_input = torch.cat([embeddings, context], 2)
        
        rnn_output, last_hidden_cell_states = self.lstm(
            rnn_input, 
            (last_hidden_cell_states[0],
             last_hidden_cell_states[1])
        )

        # Concatenate LSTM output, context and embeddings 
        # and feed to the output layer
        final_features = torch.cat([rnn_output, 
                                    context, 
                                    embeddings], -1).squeeze(1)
        out_proj = self.out_transform(final_features)

        return out_proj, last_hidden_cell_states

    
class TransformeRNN(torch.nn.Module):
    
    """
    PyTorch module that implements an encoder-decoder 
    model for sequence-to-sequence task

    Args:
        src_vocabulary_size (int): 
            The size of the source vocabulary.
        tgt_vocabulary_size (int): 
            The size of the target vocabulary.
        config (dict): 
            Dictionary from which model hyperparameters 
            are to be extracted, next keys matters:
                embedding_size (int): 
                    The size of the word embeddings 
                    used by the encoder and decoder
                hidden_size (int): 
                    The number of units in the encoder
                    and decoder LSTM layers
                dropout (float, optional): 
                    The probability of dropping out units 
                    in the encoder and decoder LSTM layers.
    """

    def __init__(self, 
                 src_vocabulary_size: int, tgt_vocabulary_size: int, 
                 config: dict) -> None:

        super(TransformeRNN, self).__init__()
        
        self.src_vocabulary_size = src_vocabulary_size
        self.tgt_vocabulary_size = tgt_vocabulary_size
        
        self.encoder = Encoder(src_vocabulary_size=src_vocabulary_size, 
                               embedding_size=config['embedding_size'], 
                               hidden_size=config['hidden_size'], 
                               dropout=config['embedding_dropout'])
        
        self.decoder = Decoder(tgt_vocabulary_size=tgt_vocabulary_size, 
                               embedding_size=config['embedding_size'], 
                               hidden_state_size=config['hidden_size'], 
                               dropout=config['embedding_dropout'])
        
        self.device = torch.device(config['device'])
        self.to(self.device)

    def forward(self, 
                src_ids: torch.tensor, 
                src_lengths: list, 
                tgt_ids: torch.tensor, 
                teacher_forcing_ratio: float = 0.5) -> torch.tensor:
        """
        This function performs one forward 
        pass through the encoder-decoder model.
        
        Args:
            src_ids (torch.tensor): 
                The source sequence tensor of shape (batch_size, seq_len).
            src_lengths (list): 
                A list of integers, where item i is the
                length of the i-th source sequence in batch.
            tgt_ids (torch.tensor): 
                The target sequence tensor of shape (batch_size, seq_len).
            teacher_forcing_ratio (float): 
                A float representing the probability of teacher forcing.
        
        Returns:
            A tensor containing the decoder output.
        """
        src_ids = src_ids.to(self.device)
        tgt_ids = tgt_ids.to(self.device)

        # pass the input sequence and their lengths through the encoder
        encoder_output, last_hidden_state = self.encoder(src_ids, src_lengths)

        batch_size, max_sequence_len = tgt_ids.shape
        tgt_vocabulary_size = self.decoder.get_embeddings.weight.shape[0]
        outputs = torch.zeros(batch_size, 
                              max_sequence_len, 
                              tgt_vocabulary_size).to(self.device)
        
        padding_src_mask = (src_ids == PAD_IDX).to(self.device)

        initial_cell_state = torch.zeros(last_hidden_state.shape).to(self.device)
        last_hidden_cell_states = (last_hidden_state, initial_cell_state)
        output = torch.full((batch_size, 1), BOS_IDX).to(self.device)

        for t in range(max_sequence_len):

            # do a forward pass through the decoder
            output, last_hidden_cell_states = self.decoder(
                tgt_ids=output,
                last_hidden_cell_states=last_hidden_cell_states,
                encoder_outputs=encoder_output,
                padding_src_mask=padding_src_mask
            )

            outputs[:, t] = output
            # either use for next prediction reference or predicted token_id
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = (tgt_ids[:, t] if is_teacher else top1).unsqueeze(1)

        return outputs


    
