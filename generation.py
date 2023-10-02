"""Generation functions for model inference."""

import heapq
from typing import List, Tuple

import numpy as np
import torch
from dataclasses import dataclass, field
from general_utils import BOS_IDX, EOS_IDX
from preprocess_tools import Tokenizer
from dataclasses_utils import get_dataloader

@dataclass
class BSNode():
    score: float = None
    base_ids: list = None
    hypothetical_id: float = None
    hidden_state: torch.Tensor = None
    cell_state: torch.Tensor = None

    def __lt__(self, other):
        return self.score < other.score

def get_new_item(base_item: BSNode, candidate: tuple) -> None:

    base_hypothesis_score = base_item.score
    cur_ids = base_item.base_ids + [base_item.hypothetical_id]
    cur_hypo_len = len(cur_ids)
    pair_score = float(candidate[0])
    pair_id = int(candidate[1])

    # Create a new item from the current hypothesis and the new candidate
    base_denormed_score = base_hypothesis_score * np.sqrt(cur_hypo_len)
    new_score = (base_denormed_score - pair_score) 
    new_normed_score = new_score / np.sqrt(cur_hypo_len + 1)
    new_item = BSNode(
        score=new_normed_score,
        base_ids=cur_ids,
        hypothetical_id=pair_id
    )
    return new_item


def beamsearch_generation(model: torch.nn.Module,
                          src_seq: str,
                          src_tokenizer: Tokenizer,
                          tgt_tokenizer: Tokenizer,
                          max_seq_len: int = 50,
                          beamsize: int = 5,
                          n_hypos: int = 5) -> List[Tuple]:
    """
    This function generates a sequence 
    of text using a pre-trained encoder-decoder model.

    Args:
        src_seq (torch.tensor): 
            Source string which will be preprocessed and fed into model's encoder.
        max_seq_len (int): 
            The maximum length that the generated sequence can have.
        beamsize (int): 
            The beamsize to be used for beam search.
        n_hypos (int): 
            The number of hypothesis that will be returned as output.

    Returns:
        A list containing the top n_hypos hypothesis.
    """
    model.eval()

    # Disable gradient tracking since we 
    # will only use the model for inference
    with torch.no_grad():
        
        src_tokenized = src_tokenizer.tokenize_doc(src_seq)
        src_encoded = src_tokenizer.encode_doc(src_tokenized)
        inference_dataloader = get_dataloader([src_encoded], batch_size=1, drop_last=False)
        packed_seq = next(iter(inference_dataloader))
        
        # Pass the source sequence through the encoder
        encoder_output, last_hidden_state = model.encoder(packed_seq['src_ids'],
                                                          packed_seq['src_lengths'])

        # Initialize the beam search starting item
        cur_ids = BOS_IDX
        cur_item = {
            'score': 0,
            'base_ids': [],
            'hypothetical_id': cur_ids,
            'hidden_state': last_hidden_state,
            'cell_state': torch.zeros(last_hidden_state.shape)
        }
        
        final_hypotheses = []
        partial_hypotheses = [cur_item]
        
        # Start the beam search loop
        while partial_hypotheses:
            cur_item = heapq.heappop(partial_hypotheses)

            model_input_id = torch.tensor([cur_item['hypothetical_id']])
            model_input_id.unsqueeze_(0)
            
            last_hidden_cell_states = (cur_item['hidden_state'],
                                       cur_item['cell_state'])

            # Pass the current item through the decoder
            logits, last_hidden_cell_states = model.decoder(
                model_input_id,
                last_hidden_cell_states,
                encoder_output)

            # Compute the log probabilities and select the top k candidates
            logps = torch.log_softmax(logits, -1).squeeze(0)
            topk_pairs = logps.topk(beamsize)

            cur_hypo_len = len(cur_item['base_ids'])
            # Create new items from the topk candidates
            for candidate in zip(topk_pairs.values, topk_pairs.indices):
                new_item = get_new_item(cur_item, candidate)                
                new_item['hidden_state'] = last_hidden_cell_states[0]
                new_item['cell_state'] = last_hidden_cell_states[1]
                
                # Check if the new item is a final hypothesis
                if new_item['hypothetical_id'] == EOS_IDX \
                        or cur_hypo_len >= max_seq_len:
                    final_hypotheses.append((new_item[0], new_item[1][0]))
                else:
                    heapq.heappush(partial_hypotheses, new_item)

            # Prune the partial hypotheses to match the beamsize
            if len(partial_hypotheses) > beamsize:
                partial_hypotheses = heapq.nsmallest(beamsize,
                                                     partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        # Sort the final hypotheses and return the top n_hypos
        res = sorted(final_hypotheses)
        return res[:n_hypos]
