"""Functions for text data preprocessing, vocabulary building."""

from typing import List, Dict, Union
from collections import defaultdict, OrderedDict
from general_utils import (SEQ_TYPES, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)


def dict2dict(foo):
    """
    Decorator for somewhat function polymorphism. If passed data type is
    dict, perform passed function map-like execution on each dict item and return 
    dict with the same keys. Else list type is expected, and no processing is required.
    """
    def _wrapper(data: Union[dict , list], **kwargs):
        if isinstance(data, dict):
            processed_dict = dict()
            for seq_type in SEQ_TYPES:
                data_chunk = data[seq_type]
                processed_chunk = foo(data_chunk, **kwargs)
                processed_dict[seq_type] = processed_chunk
            return processed_dict
        else:
            processed_data = foo(data, **kwargs)
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
    data = dict()    
    # read two files contains SRC and TGT texts
    for seq_type in SEQ_TYPES:
        # open file according to the type of sequences needed (SRC or TGT)
        with open(path + filename[seq_type]) as fstream:
            data[seq_type] = fstream.readlines()
    return data


# TODO: branch model pipeline with Tokenizer and without
class Tokenizer(object):
    def __init__(self) -> None:
        self.vocabulary = None

    def _get_data_chunk(data: Union[dict, list]) -> list:
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
    
    # TODO: add courpus tokenize function
    # Define a function that tokenizes the input string
    def tokenize_doc(doc: str, skip_special_tokens: bool = False) -> List[str]:
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
        doc = [token for token in doc.split()]

        # If `skip_special_tokens` is `False`, add two 
        # special tokens to the beginning and end of the tokenized document
        if not skip_special_tokens:
            doc = [BOS_TOKEN] + doc + [EOS_TOKEN]

        # Return the tokenized document as a list of tokens
        return doc
    
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
 

        for token in tokenized_doc:
            self.vocabulary[token]
        
        return doc
    
    @dict2dict
    def encode_texts(self,
                     tokenized_data: list, 
                     skip_special_tokens: bool = False) -> Dict:
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
        # TODO: change documentation
        encoded_texts = dict()
        
        corpus = []
        # produce encoding for SRC and TGT sequences
        for doc in tokenized_data:
            tokenized_doc = []
            for token in doc:
                tokenized_doc.append(self.vocabulary[token])
            corpus.append(tokenized_doc)
        # TODO: add decorator for dict/list preproces    
        return encoded_texts
        
    def build_vocabulary(self, data: Union[dict, list]) -> None:
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
        for data_chunk in self._get_data_chunk(data):
            for doc in data_chunk:
                # tokenize and produce iteration by each token
                for token in doc.split():
                    # if vocabulary contains token, we increment value assigned 
                    # to the token, else we add new key and assign 1 to it
                    token_occ_counter[token] += 1
        # sort counted tokens in decreasing order and append them to special tokens
        special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        sorted_keys = special_tokens + sorted(
            token_occ_counter, 
            key=lambda x: (token_occ_counter[x], x), 
            reverse=True)
    
        # assign id for each token according to their sorted position
        token_indexes = [i for i in range(len(sorted_keys))]
        self.vocabulary = OrderedDict(zip(sorted_keys, token_indexes))


def build_vocabulary(data: dict) -> dict:
    """
    This function takes dict of two SRC and TGT lists of rows. 
    Performes basic split tokenization on each row and returns 
    vocabulary dict with sorted tokens according to their number of occurences

    Args:
        data (dict): 
            Dict which contains two lists of 
            SRC and TGT untokenized string rows

    Returns:
        vocabulary (OrderedDict): 
            Dict which contains tokens as keys 
            and their ids as values sorted based 
            on tokens occurence frequency.
    """
    # dict for token occurence counting, initialized with 0 
    # for each new key and thus available for incrementation
    token_occ_counter = defaultdict(int)
    
    # perform token enumeration for SRC and TGT sequences
    for seq_type in SEQ_TYPES:
        # use yield
        for doc in data[seq_type]:
            # tokenize and produce iteration by each token
            for token in doc.split():
                # if vocabulary contains token, we increment value assigned 
                # to the token, else we add new key and assign 1 to it
                token_occ_counter[token] += 1
    # sort counted tokens in decreasing order and append them to special tokens
    special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
    sorted_keys = special_tokens + sorted(
        token_occ_counter, 
        key=lambda x: (token_occ_counter[x], x), 
        reverse=True)
    
    # assign id for each token according to their sorted position
    token_indexes = [i for i in range(len(sorted_keys))]
    vocabulary = OrderedDict(zip(sorted_keys, token_indexes))
    return vocabulary


# Define a function that tokenizes the input string
def doc_tokenizer(doc: str, skip_special_tokens: bool = True) -> List[str]:
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
    doc = [token for token in doc.split()]

    # If `skip_special_tokens` is `False`, add two 
    # special tokens to the beginning and end of the tokenized document
    if not skip_special_tokens:
        doc = [BOS_TOKEN] + doc + [EOS_TOKEN]

    # Return the tokenized document as a list of tokens
    return doc


def encode_texts(vocabulary: Dict, 
                 data: Dict, 
                 skip_special_tokens: bool = False) -> Dict:
    """
    Function accepts vocabulary and dictionary of 
    untokenized list of texts and returns dictionary
    of tokenized and encoded texts

    Args:
        vocabulary (Dict): 
            Pre-made vocabulary contains tokens as keys and token ids as values
        data (Dict): 
            Dictionary of source and target lists of untokenized texts.

    Returns:
        encoded_texts (Dict): 
            Dictionary of target and sequence texts tokenized and encoded 
            with vocabulary.
    """
    encoded_texts = dict()
    
    corpus = []
    # produce encoding for SRC and TGT sequences
    for seq_type in SEQ_TYPES:
        for doc in data[seq_type]:
            doc = doc_tokenizer(doc=doc, skip_special_tokens=skip_special_tokens)
            tokenized_doc = []
            for token in doc:
                tokenized_doc.append(vocabulary[token])
            corpus.append(tokenized_doc)
        encoded_texts[seq_type] = corpus
     
    return encoded_texts
