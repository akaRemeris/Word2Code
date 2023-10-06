"""Functions for text data preprocessing, vocabulary building."""

from typing import List, Dict, Union, Callable
from collections import defaultdict, OrderedDict
from general_utils import (SEQ_TYPES, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, UNK_IDX)


def dict2dict(foo2wrap):
    """
    Decorator for somewhat function polymorphism. If passed data type is
    dict, perform passed function map-like execution on each dict item and return 
    dict with the same keys. Else list type is expected, and no processing is required.
    """
    def _wrapper(self, data: Union[dict , list], **kwargs):
        if isinstance(data, dict):
            processed_dict = {}
            for seq_type in SEQ_TYPES:
                data_chunk = data[seq_type]
                processed_chunk = foo2wrap(self, data_chunk, **kwargs)
                processed_dict[seq_type] = processed_chunk
            return processed_dict
        processed_data = foo2wrap(self, data, **kwargs)
        return processed_data
    return _wrapper


def read_src_tgt_dataset(path: str, filename: dict) -> Dict:
    """
    Function performes reading of two separate 
    files which contain SRC and TGT texts.

    Args:
        path (str): 
            Path to the datasets directory
        filename (dict): 
            Contains filename as values according to the key either SRC or TGT

    Returns:
        data (dict): 
            Dictionary with SRC and TGT keys and lists of string rows as values
    """
    data = {}
    # read two files contains SRC and TGT texts
    for seq_type in SEQ_TYPES:
        # open file according to the type of sequences needed (SRC or TGT)
        with open(path + filename[seq_type], encoding='utf-8') as fstream:
            data[seq_type] = fstream.readlines()
    return data

def default_tokenization(doc: str, skip_special_tokens: bool = False) -> List[str]:
    """
    This function takes in a string of document and a boolean 
    variable `skip_special_tokens` that tells whether to 
    skip special tokens or not. It tokenizes the input document
    by splitting it into individual words. If `skip_special_tokens`
    is `False`, it adds two special tokens; BOS_TOKEN at the beginning 
    and EOS_TOKEN at the end of the tokenized document using list notation []. 
    Finally, it returns the tokenized document as a list of tokens.

    Args:
        doc (str): 
            A string of document
        skip_special_tokens (bool): 
            A boolean variable that represents whether
            to skip special tokens or not. Default value is True.
        
    Returns:
        doc (list): 
            List of str tokens
    """

    # Tokenize the input document by splitting it into individual words
    doc = list(doc.split())

    # If `skip_special_tokens` is `False`, add two
    # special tokens to the beginning and end of the tokenized document
    if not skip_special_tokens:
        doc = [BOS_TOKEN] + doc + [EOS_TOKEN]

    # Return the tokenized document as a list of tokens
    return doc


# TODO: add documentation
class Tokenizer():
    def __init__(self,
                 tokenization_foo: Callable[[str, bool], list]=default_tokenization) -> None:
        self.vocabulary = None
        self.reversed_vocabulary = None
        self.tokenize_doc = tokenization_foo


    def _get_data_chunk(self, data: Union[dict, list]) -> list:
        """
        Yields data elements depending on type of data.
        If data is a dict containing two types of sequences (lists) as values,
        chunker will iteratively return each chunks in order defined in SEQ_TYPES,
        otherwise (if data is list already) data returns as it is.

        Args:
            data (dict | list): 
                List or dict of lists, those lists represent chunks of string docs.
            
        Yields:
            data (list): Yields chunks represented as list of strings.
        """
        if isinstance(data, dict):
            for key in SEQ_TYPES:
                yield data[key]
        else:
            yield data

    #TODO: Define a function that tokenizes the input string
    
    @dict2dict
    def tokenize_corpus(self,
                        corpus: list,
                        skip_special_tokens: bool = False):
        tokenized_corpus = []
        for doc in corpus:
            tokenized_doc = self.tokenize_doc(doc, skip_special_tokens)
            tokenized_corpus.append(tokenized_doc)
        return tokenized_corpus
    
    def encode_doc(self, tokenized_doc: List[str]) -> List[int]:
        encoded_doc = []
        for token in tokenized_doc:
            encoded_doc.append(self.vocabulary.get(token, UNK_IDX))
        return encoded_doc
    
    def decode_doc(self, encoded_doc: List[str]) -> List[int]:
        if self.reversed_vocabulary is None:
            self.reversed_vocabulary = dict(map(reversed, self.vocabulary.items()))
        decoded_doc = []
        for token_id in encoded_doc:
            token = self.reversed_vocabulary[token_id]
            if token is EOS_TOKEN:
                break
            decoded_doc.append(token)
        return decoded_doc
    
    @dict2dict
    def encode_corpus(self, tokenized_data: list) -> Dict:
        """
        Function accepts vocabulary and dictionary of 
        untokenized list of texts and returns dictionary
        of tokenized and encoded texts

        Args:
            data (Dict):
                Dictionary of source and target lists of untokenized texts.

        Returns:
            encoded_texts (Dict): 
                Dictionary of target and sequence texts tokenized and encoded 
                with vocabulary.
        """
        # TODO: change documentation with decorator information

        encoded_corpus = []
        # produce encoding for SRC and TGT sequences
        for doc in tokenized_data:
            encoded_corpus.append(self.encode_doc(doc))
        return encoded_corpus

    def build_vocabulary(self,
                         tokenized_data: Union[dict, list],
                         max_token_freq: float = 2,
                         min_token_count: int = 0) -> None:
        """
        This function takes dict of SRC and TGT lists of strings. 
        Performs basic split tokenization on each row and initialize 
        tokenizer object's vocabulary dict with sorted tokens according 
        to their number of occurences.

        Args:
            data (dict | list): 
                List or dict of lists, those lists represent chunks of string docs.

        """
        # dict for token occurence counting, initialized with 0
        # for each new key and thus available for incrementation
        token_occ_counter = defaultdict(int)

        # iterate through all list or dict data chunks
        for data_chunk in self._get_data_chunk(tokenized_data):
            doc_counter = 0
            for tokenized_doc in data_chunk:
                doc_counter += 1
                # tokenize and produce iteration by each token
                for token in set(tokenized_doc):
                    # if vocabulary contains token, we increment value assigned
                    # to the token, else we add new key and assign 1 to it
                    token_occ_counter[token] += 1
        # sort counted tokens in decreasing order and append them to special tokens
        special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        # reset special tokens counters
        for special_token in special_tokens:
            try:
                token_occ_counter.pop(special_token)
            except KeyError:
                pass

        filtered_occurances = {
            token: n_occurances for token, n_occurances in token_occ_counter.items()
            if n_occurances/doc_counter <= max_token_freq
            and n_occurances >= min_token_count
        }

        sorted_keys = special_tokens + sorted(
            filtered_occurances,
            key=lambda x: (filtered_occurances[x], x),
            reverse=True)

        # assign id for each token according to their sorted position
        token_indexes = list(range(len(sorted_keys)))
        self.vocabulary = OrderedDict(zip(sorted_keys, token_indexes))
