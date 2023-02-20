import numpy as np


def get_lines(file_path):
    """
    given the path to the dataset it returns it in format
    [['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], ...]

    with the respective labels
    [['B-ORG\n', 'O\n', 'B-MISC\n', 'O\n', 'O\n', 'O\n', 'B-MISC\n', 'O\n', 'O\n'], ...]

    :param file_path: (str) path to the file
    :return: (list. list) list of lists where each element is a sentence and each element of
    the sublist is a word in that sentence and respective list of lists with a
    label for each word
    """

    with open(file_path, mode='rt', encoding="UTF8") as f:

        labels_sentence = []  # list of labels of a single sentence
        labels = []  # list of lists with all the sentences
        sentences = []
        sentence = []
        for line in f:
            if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                if len(sentence) > 0 and len(labels_sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
                    labels.append(labels_sentence)
                    labels_sentence = []
                continue
            splits = line.split(' ')
            sentence.append(splits[0])
            labels_sentence.append(splits[-1])

    if len(sentence) > 0 and len(labels_sentence) > 0:
        sentences.append(sentence)
        labels.append(labels_sentence)

    return sentences, labels


def get_char_info(sentences):
    """
    add to the sentences list the character-level info for CNN embedding in a format
    [[['EU', ['E', 'U']], ['rejects', ['r', 'e', 'j', 'e', 'c', 't', 's']], ...]
    :param sentences: (list) the list of lists with the sentences
    :return: (list) list of lists where each word is paired with the carachter-level info
    """

    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            chars = [c for c in word]
            sentences[i][j] = [word, chars]

    return sentences


def get_dataset(file_path):

    sentences, labels = get_lines(file_path)
    sentences_charinfo = get_char_info(sentences)

    return sentences_charinfo, labels


def get_casing(word, case_lookup):
    """
    given a dictionary-type lookup table that maps the different casing of a word
    to a set of indices it returns the aforementioned index
    :param word: (str)
    :param case_lookup: (dict) dictionary that given a casing type it returns an index
    :return: (int) the casing index of the 'word' variable
    """

    casing = 'other'
    num_digits = 0
    for char in word:
        if char.isdigit():
            num_digits += 1

    digit_fraction = num_digits / float(len(word))

    if word.isdigit():  # is a digit
        casing = 'numeric'
    elif digit_fraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower():  # all lower case
        casing = 'allLower'
    elif word.isupper():  # all upper case
        casing = 'allUpper'
    elif word[0].isupper():  # is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif num_digits > 0:
        casing = 'contains_digit'

    return case_lookup[casing]

