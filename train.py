import preprocessing
import architecture
import tensorflow as tf
from tensorflow import keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


lr = 1e-4
epochs = 1
batch_size = 64

train_data_list, val_data_list, labels_train, labels_val = preprocessing.get_dataset('train_data.txt',
                                                                                     'val_data.txt',
                                                                                     'glove.6B.50d.txt')
word2idx,\
    case2idx,\
    char2idx,\
    label2idx,\
    word_embeddings,\
    case_embeddings = preprocessing.get_dicts_and_embeddings('data/train_data.txt',
                                                             'data/val_data.txt',
                                                             'glove.6B.50d.txt')

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

history = model.fit(x=train_data_list,
                    y=labels_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(val_data_list, labels_val),
                    callbacks=callbacks)
