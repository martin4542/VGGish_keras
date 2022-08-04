import vggish_input
import vggish_params as params
from vggish import VGGish
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout
from keras.engine.topology import get_source_inputs
from keras import backend as K

import os
import librosa
import numpy as np
import tensorflow as tf
from random import shuffle
from scipy.signal import butter, lfilter

audio_path = 'C:/Users/jae/Music/hearing_loss_split'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:   
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
        
# define vggish + mlp model

def VGGish_MLP(load_weights=True, weights='audioset',
               input_tensor=None, input_shape=None, out_dim=None,
               include_top=True, pooling='avg', num_of_label=3):
    
    if out_dim is None: out_dim = params.EMBEDDING_SIZE
    if input_shape is None: input_shape = (params.NUM_FRAMES, params.NUM_BANDS, 1)
    if input_tensor is None:
        aud_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            aud_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            aud_input = input_tensor
    
    audio_feature_extractor = VGGish(load_weights=load_weights, weights=weights,
                                     input_tensor=input_tensor, input_shape=input_shape, out_dim=out_dim,
                                     include_top=include_top, pooling=pooling)
    aud_feat = audio_feature_extractor(aud_input)
    x = Dense(128, activation='relu', name='mlp/fc1')(aud_feat)
    #x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', name='mlp/fc2')(x)
    #x = Dropout(0.4)(x)
    x = Dense(num_of_label, activation='sigmoid', name='mlp/output')(x)
    
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = aud_input
    
    model = Model(inputs, x, name='VGGish_mlp')
    
    return model

# for preprocessing signals - low pass filtering
def butter_lowpass(cut, fs, order=5):
    nyq = 0.5 * fs
    cut = cut / nyq
    return butter(order, cut, btype='low')

def butter_lowpass_filter(data, cut, fs, order=9):
    b, a = butter_lowpass(cut, fs, order=order)
    return lfilter(b, a, data)

# log-mel feature extraction
# (n, 96, 64) shape log-mel spectrogram is input of vggish
def get_vggish_input(category='normal'):
    feature_len = np.empty((0,))
    log_mel_feature = np.empty((0, 96, 64))
    for folder in os.listdir(os.path.join(audio_path, category)):
        for audio_file in os.listdir(os.path.join(audio_path, category, folder)):
            #if audio_file.endswith('.wav') and (audio_file.lower() == 'chapter1.wav' or audio_file.lower() == 'chapter2.wav'):
            #if audio_file.endswith('.wav') and audio_file.lower() == 'chapter3.wav':
            if audio_file.endswith('.wav') and (audio_file.lower() == 'chapter7.wav' or audio_file.lower() == 'chapter8.wav'):
                x, sr = librosa.load(os.path.join(audio_path, category, folder, audio_file), sr=44100, mono=True) # load data
                #x = butter_lowpass_filter(x, cut=1000, fs=sr) # preprocessing - low pass filtering
                temp = vggish_input.waveform_to_examples(x, sr) # get log-mel feature
                log_mel_feature = np.append(log_mel_feature, temp, axis=0)
        feature_len = np.append(feature_len, [len(log_mel_feature)])
    return log_mel_feature, feature_len

# get log-mel feature of normal, mild, severe
# normal is class index 0, mild 1, severe 2
normal_feat, normal_feat_len = np.array(get_vggish_input(category='normal'))
mild_feat, mild_feat_len = np.array(get_vggish_input(category='mild'))
severe_feat, severe_feat_len = np.array(get_vggish_input(category='severe'))

mild_feat_len += normal_feat_len[-1]
severe_feat_len += mild_feat_len[-1]
total_feat_len = np.append(normal_feat_len, mild_feat_len)
total_feat_len = np.append(total_feat_len, severe_feat_len)


# for 3 class classification
"""
normal_labels = np.array([[1,0,0]] * normal_feat.shape[0])
mild_labels = np.array([[0,1,0]] * mild_feat.shape[0])
severe_labels = np.array([[0,0,1]] * severe_feat.shape[0])

X = np.concatenate((normal_feat, mild_feat, severe_feat))
Y = np.concatenate((normal_labels, mild_labels, severe_labels))
"""
# for 2 class classification
normal_labels = np.array([[1,0]] * (normal_feat.shape[0]))
hr_labels = np.array([[0,1]] * (mild_feat.shape[0] + severe_feat.shape[0]))

X = np.concatenate((normal_feat, mild_feat, severe_feat))
Y = np.concatenate((normal_labels, hr_labels))

print(X.shape, Y.shape)

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

NUM_OF_FOLD = 5
NUM_CLASSES = 3
NUM_BATCHES = 30
NUM_EPOCHS = 20

val_acc = []
for i in range(NUM_OF_FOLD):
    index = [k for k in range(len(X))]
    test_idx = []
    for j in range(len(total_feat_len)):
        if j >= 1:
            seq_len = int(total_feat_len[j]-total_feat_len[j-1])
            temp = [k for k in range(int(total_feat_len[j-1]+seq_len*i*0.2), int(total_feat_len[j-1]+seq_len*(i+1)*0.2))]
            test_idx.extend(temp)
        else:
            seq_len = int(total_feat_len[j])
            temp = [k for k in range(int(seq_len*i*0.2), int(seq_len*(i+1)*0.2))]
            test_idx.extend(temp)
    train_idx = np.delete(index, test_idx)
    shuffle(train_idx)
    
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]
    
    vggish_mlp = VGGish_MLP(num_of_label=2)
    adam = Adam(learning_rate=0.001)
    vggish_mlp.compile(optimizer=adam,
                    loss = 'binary_crossentropy',
                    #loss='categorical_crossentropy',
                    metrics=['accuracy'])
    callback_list = [ModelCheckpoint(filepath='checkpoints/fold_'+str(i+1)+'vggish_mlp.h5', 
                                     monitor='val_accuracy')]#, save_best_only=True)]
    vggish_mlp.summary()
    history = vggish_mlp.fit(X_train, Y_train, epochs=NUM_EPOCHS, batch_size=NUM_BATCHES,
                     validation_data=(X_test, Y_test), callbacks=callback_list, verbose=2)
    val_acc.append(max(history.history['val_accuracy']))
    K.clear_session()

for i in range(len(val_acc)):
    print(f'fold {i+1} top val acc:{val_acc[i]*100}')
print(f'average:{np.mean(val_acc)*100}')