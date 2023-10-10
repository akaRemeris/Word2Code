import re

def tokenize_src_doc(text, skip_special_tokens: bool = False):
    """
    Tokenizes question from a string into a list of strings (tokens) and reverses it
    """
    return list(filter(lambda x: len(x) < 16, re.findall(r"[\w']+", text.lower())[::-1]))

def tokenize_tgt_doc(text, skip_special_tokens: bool = False):
    """
    Tokenizes code snippet into a list of operands
    """
    return list(filter(lambda x: len(x) < 10, re.findall(r"[\w']+|[.,!?;:@~(){}\[\]+-/=\\\'\"\`]", text.lower())))