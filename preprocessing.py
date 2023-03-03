import tensorflow as tf
import numpy as np
import string


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
            labels_sentence.append(splits[-1].rstrip('\n'))

    if len(sentence) > 0 and len(labels_sentence) > 0:
        sentences.append(sentence)
        labels.append(labels_sentence)

    return sentences, labels


def get_char_info(sentences):
    """
    add to the sentences list the character-level info for CNN embedding in a format
    [[['EU', ['E', 'U']], ['rejects', ['r', 'e', 'j', 'e', 'c', 't', 's']], ...]

    :param sentences: (list) the list of lists with the sentences
    :return: (list) list of lists where each word is paired with the character-level info
    """

    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            chars = [c for c in word]
            sentences[i][j] = [word, chars]

    return sentences


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


def get_embeddings(sentences, labels, word_embedding_file):
    """
    :param sentences: (list) list of sentences in the dataset
    :param labels: (list) list of labels paired with the list of sentences
    :param word_embedding_file: (str) path to the pretrained word embedding .txt file
    :return: (dict, dict, dict, np.array, np.array) 3 dictionaries mapping words, casing, chars to
    integer indices and 2 lists containing the word_embeddings (given by the file) and the
    case_embeddings (given by a lookup table)
    """

    words = {}

    # save unique words and labels in the dataset
    for sentence, sentence_labels in zip(sentences, labels):
        for token, char in sentence:
            words[token.lower()] = True

    # dict mapping labels to integer indices
    label2idx = {'B-ORG': 1,
                 'O': 2,
                 'B-MISC': 3,
                 'B-PER': 4,
                 'B-LOC': 5,
                 'I-PER': 6,
                 'I-ORG': 7,
                 'I-LOC': 8,
                 'I-MISC': 9,
                 'PADDING': 0}

    # dict mapping casing type to integer indices
    case2idx = {'PADDING': 0,
                'allLower': 1,
                'allUpper': 2,
                'initialUpper': 3,
                'other': 4,
                'mainly_numeric': 5,
                'contains_digit': 6,
                'numeric': 7}

    # in the article the case embedding is done with a lookup table so
    # the casing information in embedded as one-hot vectors
    case_embeddings = np.identity(len(case2idx), dtype='int32')

    chars = string.ascii_letters+string.digits+string.punctuation

    # dict mapping chars to integer indices
    char2idx = {"PADDING": 0,
                "UNKNOWN": 1}
    for c in chars:
        char2idx[c] = len(char2idx)

    embedding_file = open(word_embedding_file, encoding="utf-8")

    # to be filled with the word vectors given in the embedding_file
    word_embeddings = []

    # dict mapping words to integer indices
    word2idx = {}

    for line in embedding_file:
        splits = line.strip().split(' ')
        word = splits[0]

        if len(word2idx) == 0:
            word2idx['PADDING'] = 0
            word2idx['UNKNOWN'] = 1
            vector = np.zeros(len(splits)-1)
            word_embeddings.append(vector)
            vector = np.random.uniform(-0.25, 0.25, len(splits) - 1)
            word_embeddings.append(vector)

        if word.lower() in words:
            word_vector = np.array([num for num in splits[1:]])
            word_embeddings.append(word_vector)
            word2idx[word] = len(word2idx)

    word_embeddings = np.array(word_embeddings)

    return word2idx, case2idx, char2idx, label2idx, word_embeddings, case_embeddings


def create_integer_embedding(sentences, labels,  word2idx, label2idx, case2idx, char2idx):
    """
    given sentences and labels of the dataset (as strings) and the mapping dictionaries
    it returns a list that embeds the datasets as integers in order to be fed to the
    network's embedding layer, which will output embedding vectors of real numbers

    :param sentences: (list) list of lists with sentences and character-level info
    :param labels: (list) list of lists with the labels paired with each word in a sentence
    :param word2idx: (dict) dict mapping words to integer indices obtained from the vocabulary of the
    pretrained word embedding
    :param label2idx: (dict) dictionary mapping labels to integer indices
    :param case2idx: (dict) dictionary mapping casing info to integer indices
    :param char2idx: (dict) dictionary mapping character info to integer indices
    :return: (list, list, int) the list containing the dataset and labels embedded as integers
    and the number of words in the dataset that were not in the embedding vocabulary and thus
    embedded as 'unknown' words
    """

    unknown_idx = word2idx['UNKNOWN']

    word_dataset = []
    case_dataset = []
    char_dataset = []
    labels_dataset = []

    word_count = 0
    unknown_word_count = 0

    for sentence in sentences:
        word_indices = []
        case_indices = []
        char_indices = []

        for word, char in sentence:
            word_count += 1
            if word in word2idx:
                word_idx = word2idx[word]
            elif word.lower() in word2idx:
                word_idx = word2idx[word.lower()]
            else:
                word_idx = unknown_idx
                unknown_word_count += 1
            char_idx = []
            for x in char:
                char_idx.append(char2idx[x])

            word_indices.append(word_idx)
            case_indices.append(get_casing(word, case2idx))
            char_indices.append(char_idx)

        word_dataset.append(word_indices)
        case_dataset.append(case_indices)
        char_dataset.append(char_indices)

    for sentence_labels in labels:
        label_indices = []

        for label in sentence_labels:
            label_indices.append(label2idx[label])

        labels_dataset.append(label_indices)

    return word_dataset, case_dataset, char_dataset, labels_dataset, unknown_word_count


def pad_dataset(word_dataset, case_dataset, chars_dataset, labels):
    """
    :param word_dataset: (list)
    :param case_dataset: (list)
    :param chars_dataset: (list)
    :param labels: (list)
    :return: (np.array, np.array, np.array, np.array) of int32
    """
    max_len_seq = 50
    max_len_chars = 15

    word_dataset = tf.keras.utils.pad_sequences(word_dataset, maxlen=max_len_seq, padding='post')
    case_dataset = tf.keras.utils.pad_sequences(case_dataset, maxlen=max_len_seq, padding='post')

    # chars padding
    for i, sentence in enumerate(chars_dataset):
        chars_dataset[i] = tf.keras.utils.pad_sequences(sentence, maxlen=max_len_chars, padding='post')

    # sequences of chars padding
    chars_dataset = tf.keras.utils.pad_sequences(chars_dataset, maxlen=max_len_seq, padding='post')

    labels = tf.keras.utils.pad_sequences(labels, maxlen=max_len_seq, padding='post')

    return word_dataset, case_dataset, chars_dataset, labels


def get_dataset(data_path, embedding_path):
    """
    :param data_path: (str) path to the dataset file
    :param embedding_path: (str) path to the embedding file
    :return: (np.array, no.array, np.array, np.array) padded dataset of int32 arrays
    """

    x, y = get_lines(data_path)
    x = get_char_info(x)

    word2idx, \
        case2idx, \
        char2idx, \
        label2idx, \
        word_embeddings, \
        case_embeddings = get_embeddings(x, y, embedding_path)

    word_dataset, \
        case_dataset, \
        chars_dataset, \
        labels_embedding, \
        unknown_word_count = create_integer_embedding(x,
                                                      y,
                                                      word2idx,
                                                      label2idx,
                                                      case2idx,
                                                      char2idx)

    word_dataset, \
        case_dataset, \
        chars_dataset, \
        labels = pad_dataset(word_dataset,
                             case_dataset,
                             chars_dataset,
                             labels_embedding)

    return word_dataset, case_dataset, chars_dataset, labels


def get_dicts_and_embeddings(data_path, embedding_path):
    """
    :param data_path: (str)
    :param embedding_path: (str)
    :return: (dict, dict, dict, dict, np.array, np.array)
    """

    x, y = get_lines(data_path)
    x = get_char_info(x)

    word2idx, \
        case2idx, \
        char2idx, \
        label2idx, \
        word_embeddings, \
        case_embeddings = get_embeddings(x, y, embedding_path)

    return word2idx, case2idx, char2idx, label2idx, word_embeddings, case_embeddings







