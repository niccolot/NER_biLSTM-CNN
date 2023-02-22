from tensorflow import keras
from keras import layers


class CharacterEmbedding(layers.Layer):
    def __init__(self, char2idx, embedding_dim=25,  name='Character_embedding'):
        super().__init__(name=name)
        self.char2idx = char2idx
        self.embedding_dim = embedding_dim

    def call(self, inputs):
        embedded_chars = layers.TimeDistributed(
            layers.Embedding(input_dim=len(self.char2idx),
                             output_dim=self.embedding_dim,
                             embeddings_initializer=
                             keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)))(inputs)

        return embedded_chars


class WordEmbedding(layers.Layer):
    def __init__(self, word2idx, word_embeddings, name='Word_embedding'):
        super().__init__(name=name)
        self.word2idx = word2idx
        self.word_embeddings = word_embeddings

    def call(self, inputs):
        embedded_words = layers.Embedding(input_dim=self.word_embeddings.shape[0],
                                          output_dim=self.word_embeddings.shape[1],
                                          embeddings_initializer=[self.word_embeddings],
                                          trainable=False)(inputs)

        return embedded_words




