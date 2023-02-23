from tensorflow import keras
from keras import layers


class CharacterEmbedding(layers.Layer):
    def __init__(self, char2idx, drop_rate, embedding_dim=25, conv_filters=25,  name='Character_embedding'):
        super().__init__(name=name)
        self.char2idx = char2idx
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim
        self.conv_filters = conv_filters

    def call(self, inputs):
        embedded_chars = layers.TimeDistributed(
            layers.Embedding(input_dim=len(self.char2idx),
                             output_dim=self.embedding_dim,
                             embeddings_initializer=
                             keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)))(inputs)

        dropout1 = layers.Dropout(self.drop_rate)(embedded_chars)

        conv = layers.TimeDistributed(layers.Conv1D(filters=self.conv_filters,
                                                    kernel_size=3,
                                                    padding='same',
                                                    activation='tanh'))(dropout1)

        pool = layers.TimeDistributed(layers.MaxPool1D(pool_size=self.conv_filters))(conv)

        dropout2 = layers.Dropout(self.drop_rate)(pool)

        return dropout2


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


class CasingEmbedding(layers.Layer):
    def __init__(self, case2idx, case_embeddings, name='Casing_embedding'):
        super().__init__(name=name)
        self.case2idx = case2idx
        self.case_embeddings = case_embeddings

    def call(self, inputs):
        embedded_casing = layers.Embedding(input_dim=len(self.case2idx),
                                           output_dim=len(self.case2idx),
                                           embeddings_initializer=[self.case_embeddings],
                                           trainable=False)

        return embedded_casing


def build_model(word2idx, char2idx, case2idx, word_embeddings, case_embeddings):

    word_input = layers.Input(shape=(None,), dtype='int32')
    char_input = layers.Input(shape=())

