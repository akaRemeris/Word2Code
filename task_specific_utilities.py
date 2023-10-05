import re

def tokenize_question(text):
    """
    Tokenizes question from a string into a list of strings (tokens) and reverses it
    """
    return list(filter(lambda x: len(x) < 16, re.findall(r"[\w']+", text)[::-1]))

def tokenize_snippet(text):
    """
    Tokenizes code snippet into a list of operands
    """
    return list(filter(lambda x: len(x) < 10, re.findall(r"[\w']+|[.,!?;:@~(){}\[\]+-/=\\\'\"\`]", text)))