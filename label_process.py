def map_to_hiragana(romanji: str):
    mapping = {'hu': 'fu', 'si': 'shi', 'ti': 'chi', 'tu': 'tsu'}
    if romanji in mapping:
        return mapping[romanji]
    return romanji

def romanji_to_dict(S): # construct dict from Romanji series
    romanji_set = set(S)
    # romanji_list = list(set(S))
    romanji_dict = {}
    for i, romanji in enumerate(romanji_set):
        romanji_dict[romanji] = i
    return romanji_dict

