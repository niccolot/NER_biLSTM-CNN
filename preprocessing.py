import numpy as np


def get_lines(file_path):

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

    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            chars = [c for c in word]
            sentences[i][j] = [word, chars]

    return sentences


def get_dataset(file_path):

    sentences, labels = get_lines(file_path)
    sentences_charinfo = get_char_info(sentences)

    return sentences_charinfo, labels



