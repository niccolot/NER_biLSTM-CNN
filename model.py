from tensorflow import keras
from keras import layers


class CharacterEmbedding(layers.Layer):
    def __init__(self,
                 char2idx,
                 drop_rate=0.5,
                 embedding_dim=25,
                 conv_filters=25,
                 name='Character_embedding'):
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

        pool = layers.TimeDistributed(layers.MaxPool1D(pool_size=50))(conv)

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
                                          embeddings_initializer=keras.initializers.Constant(self.word_embeddings),
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
                                           embeddings_initializer=keras.initializers.Constant(self.case_embeddings),
                                           trainable=False)(inputs)

        return embedded_casing


def build_model(word2idx,
                char2idx,
                case2idx,
                label2idx,
                word_embeddings,
                case_embeddings,
                lstm_states=275,
                block_droprate=0.5):

    word_input = layers.Input(shape=(None,), dtype='int32')
    char_input = layers.Input(shape=(None, 50), dtype='int32')
    casing_input = layers.Input(shape=(None,), dtype='int32')

    embedded_words = WordEmbedding(word2idx=word2idx, word_embeddings=word_embeddings)(word_input)
    embedded_chars = CharacterEmbedding(char2idx=char2idx)(char_input)
    embedded_casing = CasingEmbedding(case2idx=case2idx, case_embeddings=case_embeddings)(casing_input)

    embedded_chars = layers.Reshape((25,1))(embedded_chars)
    concat_input = layers.Concatenate()([embedded_words, embedded_casing, embedded_chars])

    lstm = layers.Bidirectional(layers.LSTM(lstm_states,
                                            dropout=block_droprate,
                                            return_sequences=True), name='biLSTM')(concat_input)

    output = layers.TimeDistributed(layers.Dense(len(label2idx), activation='softmax'),
                                    name="Softmax_layer")(lstm)

    return keras.Model(inputs=[word_input, char_input, casing_input], outputs=output)



