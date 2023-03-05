from tensorflow import keras
from keras import layers


class CharacterEmbedding(layers.Layer):
    def __init__(self,
                 char2idx,
                 drop_rate=0.6,
                 embedding_dim=53,
                 name='Character_embedding',
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.char2idx = char2idx
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim

        self.chars_embedding_layer = layers.TimeDistributed(
            layers.Embedding(input_dim=len(self.char2idx),
                             output_dim=self.embedding_dim,
                             input_length=15,
                             embeddings_initializer=
                             keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)))

        self.dropout = layers.Dropout(self.drop_rate)
        self.convolution = layers.TimeDistributed(
            layers.Conv1D(filters=self.embedding_dim,
                          kernel_size=3,
                          padding='same',
                          activation='relu'))

        self.pooling = layers.TimeDistributed(layers.MaxPool1D(pool_size=15))
        self.flat = layers.TimeDistributed(layers.Flatten())

    def call(self, inputs):
        embedded_chars = self.chars_embedding_layer(inputs)
        dropout1 = self.dropout(embedded_chars)
        conv = self.convolution(dropout1)
        pool = self.pooling(conv)
        flat = layers.TimeDistributed(layers.Flatten())(pool)
        dropout2 = layers.Dropout(self.drop_rate)(flat)

        return dropout2

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'char2idx': self.char2idx,
            'drop_rate': self.drop_rate,
            'embedding_dim': self.embedding_dim,
            'chars_embedding_layer': self.chars_embedding_layer,
            'dropout': self.dropout,
            'convolution': self.convolution,
            'pooling': self.pooling,
            'flat': self.flat
        })

        return config


class WordEmbedding(layers.Layer):
    def __init__(self, word2idx, word_embeddings, name='Word_embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.word2idx = word2idx
        self.word_embeddings = word_embeddings
        self.in_dim = self.word_embeddings.shape[0]
        self.out_dim = self.word_embeddings.shape[1]
        self.word_embedding_layer = layers.Embedding(self.in_dim,
                                                     self.out_dim,
                                                     input_length=50,
                                                     embeddings_initializer=keras.initializers.Constant(self.word_embeddings),
                                                     trainable=False)

    def call(self, inputs):
        embedded_words = self.word_embedding_layer(inputs)

        return embedded_words

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'word2idx': self.word2idx,
            'word_embeddings': self.word_embeddings,
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
            'word_embedding_layer': self.word_embedding_layer
        })

        return config


class CasingEmbedding(layers.Layer):
    def __init__(self, case2idx, case_embeddings, name='Casing_embedding', **kwargs):
        super().__init__(name=name, **kwargs)
        self.case2idx = case2idx
        self.case_embeddings = case_embeddings
        self.case_embedding_layer = layers.Embedding(input_dim=len(self.case2idx),
                                                     output_dim=len(self.case2idx),
                                                     embeddings_initializer=keras.initializers.Constant(self.case_embeddings),
                                                     trainable=False)

    def call(self, inputs):
        embedded_casing = self.case_embedding_layer(inputs)

        return embedded_casing

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'case2idx': self.case2idx,
            'case_embeddings': self.case_embeddings,
            'case_embedding_layer': self.case_embedding_layer
        })

        return config


def build_model(word2idx,
                case2idx,
                char2idx,
                label2idx,
                word_embeddings,
                case_embeddings,
                lstm_states=275,
                droprate=0.6,
                block_droprate=0.6):

    word_input = layers.Input(shape=(50,), dtype='int32', name='word_input')
    casing_input = layers.Input(shape=(50,), dtype='int32', name='casing_input')
    char_input = layers.Input(shape=(50, 15, ), dtype='int32', name='char_input')

    embedded_words = WordEmbedding(word2idx=word2idx, word_embeddings=word_embeddings)(word_input)
    embedded_casing = CasingEmbedding(case2idx=case2idx, case_embeddings=case_embeddings)(casing_input)
    embedded_chars = CharacterEmbedding(char2idx=char2idx)(char_input)

    concat_input = layers.Concatenate()([embedded_words, embedded_casing, embedded_chars])
    concat_input = layers.Dropout(droprate)(concat_input)

    lstm = layers.Bidirectional(layers.LSTM(lstm_states,
                                            dropout=block_droprate,
                                            return_sequences=True), name='biLSTM')(concat_input)

    output = layers.TimeDistributed(layers.Dense(len(label2idx), activation='softmax'),
                                    name="Softmax_layer")(lstm)

    return keras.Model(inputs=[word_input, casing_input, char_input], outputs=output)

