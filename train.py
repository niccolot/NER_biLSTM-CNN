import preprocessing
import architecture
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


lr = 5e-5
epochs = 200
batch_size = 64


word_dataset_train, \
    case_dataset_train, \
    chars_dataset_train, \
    labels_train = preprocessing.get_dataset('train_data.txt', 'glove.6B.50d.txt')

word2idx,\
    case2idx,\
    char2idx,\
    label2idx,\
    word_embeddings,\
    case_embeddings = preprocessing.get_dicts_and_embeddings('train_data.txt', 'glove.6B.50d.txt')

word_dataset, \
    case_dataset, \
    chars_dataset, \
    labels = preprocessing.get_dataset('val_data.txt', 'glove.6B.50d.txt')

x_train_list = [word_dataset_train, case_dataset_train, chars_dataset_train]
x_val_list = [word_dataset, case_dataset, chars_dataset]


model = architecture.build_model(word2idx=word2idx,
                                 case2idx=case2idx,
                                 char2idx=char2idx,
                                 label2idx=label2idx,
                                 word_embeddings=word_embeddings,
                                 case_embeddings=case_embeddings)

opt = keras.optimizers.Adam(learning_rate=lr)
loss = keras.losses.SparseCategoricalCrossentropy()

model.compile(loss=loss, optimizer=opt)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=20, monitor="val_loss")]

model.summary()

history = model.fit(x=x_train_list,
                    y=labels_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_val_list, labels),
                    callbacks=callbacks)