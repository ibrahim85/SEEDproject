from fuzzywuzzy import fuzz

def similarity(s, t, method):
    if method == 'partial':
        return fuzz.partial_ratio(s, t)
    elif method == 'token_sort':
        return fuzz.token_sort_ratio(s, t)
    elif method == 'token_set':
        return fuzz.token_set_ratio(s, t)

