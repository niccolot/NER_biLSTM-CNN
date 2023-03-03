import tensorflow as tf
from tensorflow import keras
from keras import layers
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()


class CharacterEmbedding(layers.Layer):
    def __init__(self,
                 char2idx,
                 drop_rate=0.5,
                 embedding_dim=25,
                 name='Character_embedding'):
        super().__init__(name=name)
        self.char2idx = char2idx
        self.drop_rate = drop_rate
        self.embedding_dim = embedding_dim

    def call(self, inputs):

        embedded_chars = layers.TimeDistributed(
            layers.Embedding(input_dim=len(self.char2idx),
                             output_dim=self.embedding_dim,
                             mask_zero=False,
                             input_length=15,
                             embeddings_initializer=
                             keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)))(inputs)

        dropout1 = layers.Dropout(self.drop_rate)(embedded_chars)

        conv = layers.TimeDistributed(layers.Conv1D(filters=self.embedding_dim,
                                                    kernel_size=3,
                                                    padding='same',
                                                    activation='tanh'))(dropout1)

        pool = layers.TimeDistributed(layers.MaxPool1D(pool_size=15))(conv)

        flat = layers.TimeDistributed(layers.Flatten())(pool)

        dropout2 = layers.Dropout(self.drop_rate)(flat)

        embedded_chars = layers.TimeDistributed(layers.Flatten())(dropout2)

        return embedded_chars

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'char2idx': self.char2idx,
            'drop_rate': self.drop_rate,
            'embedding_dim': self.embedding_dim,
        })

        return config


class WordEmbedding(layers.Layer):
    def __init__(self, word2idx, word_embeddings, name='Word_embedding'):
        super().__init__(name=name)
        self.word2idx = word2idx
        self.word_embeddings = word_embeddings

    def call(self, inputs):
        embedded_words = layers.Embedding(input_dim=self.word_embeddings.shape[0],
                                          output_dim=self.word_embeddings.shape[1],
                                          input_length=50,
                                          mask_zero=True,
                                          embeddings_initializer=keras.initializers.Constant(self.word_embeddings),
                                          trainable=False)(inputs)

        return embedded_words

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'word2idx': self.word2idx,
            'word_embeddings': self.word_embeddings,
        })

        return config


class CasingEmbedding(layers.Layer):
    def __init__(self, case2idx, case_embeddings, name='Casing_embedding'):
        super().__init__(name=name)
        self.case2idx = case2idx
        self.case_embeddings = case_embeddings

    def call(self, inputs):
        embedded_casing = layers.Embedding(input_dim=len(self.case2idx),
                                           output_dim=len(self.case2idx),
                                           embeddings_initializer=keras.initializers.Constant(self.case_embeddings),
                                           trainable=False)(inputs)

        return embedded_casing

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'case2idx': self.case2idx,
            'case_embeddings': self.case_embeddings
        })

        return config


def build_model(word2idx,
                case2idx,
                char2idx,
                label2idx,
                word_embeddings,
                case_embeddings,
                lstm_states=275,
                block_droprate=0.5):

    word_input = layers.Input(shape=(50,), dtype='int32', name='word_input')
    casing_input = layers.Input(shape=(50,), dtype='int32', name='casing_input')
    char_input = layers.Input(shape=(50, 15, ), dtype='int32', name='char_input')

    embedded_words = WordEmbedding(word2idx=word2idx, word_embeddings=word_embeddings)(word_input)
    embedded_casing = CasingEmbedding(case2idx=case2idx, case_embeddings=case_embeddings)(casing_input)
    embedded_chars = CharacterEmbedding(char2idx=char2idx)(char_input)

    concat_input = layers.Concatenate()([embedded_words, embedded_casing, embedded_chars])

    lstm = layers.Bidirectional(layers.LSTM(lstm_states,
                                            dropout=block_droprate,
                                            return_sequences=True), name='biLSTM')(concat_input)

    output = layers.TimeDistributed(layers.Dense(len(label2idx), activation='softmax'),
                                    name="Softmax_layer")(lstm)

    return keras.Model(inputs=[word_input, casing_input, char_input], outputs=output)

