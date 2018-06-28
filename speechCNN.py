
'''
Simple audio classification with Keras

This code is translated following rstudio/keras code.
https://tensorflow.rstudio.com/blog/simple-audio-classification-keras.html

Environment: MacOS 10.12, Python 3.6.4, Tensorflow 1.8.0, Keras 2.2.0
'''

import os
mydir = '/Users/xxx/Documents'

###############################################################################
### Download data
import urllib.request
import tarfile

os.mkdir(mydir+'/spdat')

url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
destfile = "/speech_commands_v0.01.tar.gz"
urllib.request.urlretrieve(url, mydir+'/spdat'+destfile)

tarfile.open(mydir+'/spdat'+destfile).extractall(mydir+'/spdat/speech_commands_v0.01')

###############################################################################
### Importing
import glob
import numpy as np
import pandas as pd
import math
import re
import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops

files = glob.glob(mydir+'/spdat/speech_commands_v0.01/*/*.wav')
word = [os.path.split(os.path.split(files[x])[0])[1] for x in range(len(files))]
df = pd.DataFrame({'fname': files, 'word': word}).query("word!='_background_noise_'")

category = df['word'].unique()
word_id = df['word']
for i in range(len(category)):
    word_id = word_id.str.replace(category[i], str(i))
def remove_text(text):
    return re.sub(r'[a-z]', '', text)
word_id = word_id.apply(remove_text)
word_id.name = 'word_id'

df = pd.concat([df, word_id.astype(np.int)], axis=1).reset_index(drop=True)

###############################################################################
### Generator
def _spectrogram_function(features, labels):
    # decoding wav files
    audio_binary = tf.read_file(features)
    wav = audio_ops.decode_wav(audio_binary, desired_channels=1)

    # create the spectrogram
    spectrogram = audio_ops.audio_spectrogram(
        wav.audio,
        window_size=window_size,
        stride=stride,
        magnitude_squared=True
    )
    spectrogram = tf.log(tf.abs(spectrogram) + 0.01)
    spectrogram = tf.transpose(spectrogram, perm=[1, 2, 0])

    # transform the class_id into a one-hot encoded vector
    response = tf.one_hot(labels, 30)

    return [spectrogram, response]

def data_generator(df, batch_size=32, shuffle=True, window_size_ms=30, window_stride_ms=10):

    window_size = int(16000 * window_size_ms / 1000)
    stride = int(16000 * window_stride_ms / 1000)
    fft_size = int(2 ** math.trunc(math.log(window_size, 2)) + 1)
    n_chunks = len(range(int(window_size / 2), int(16000 - window_size / 2)+1, stride))

    ds = tf.data.Dataset.from_tensor_slices((df.fname, df.word_id))
    if shuffle:
        ds = ds.shuffle(buffer_size=100000)

    ds = ds.map(_spectrogram_function).repeat()
    ds = ds.padded_batch(
        batch_size=batch_size,
        padded_shapes=([n_chunks, fft_size, None], [None])
    )

    sess = tf.Session()
    iterator = ds.make_one_shot_iterator()
    next_batch = iterator.get_next()

    while True:
        yield sess.run(next_batch)

np.random.seed(6)
id_train = df.sample(n=int(0.7 * len(df))).index
win1 = 30
win2 = 10
bsize = 32

ds_train = data_generator(
    df[df.index.isin(id_train)],
    batch_size=bsize,
    window_size_ms=win1,
    window_stride_ms=win2
)
ds_validation = data_generator(
    df[~df.index.isin(id_train)],
    batch_size=bsize,
    shuffle = False,
    window_size_ms=win1,
    window_stride_ms=win2
)

###############################################################################
### Model definition
window_size_ms = win1
window_stride_ms = win2
window_size = int(16000 * window_size_ms / 1000)
stride = int(16000 * window_stride_ms / 1000)
fft_size = int(2 ** math.trunc(math.log(window_size, 2)) + 1)
n_chunks = len(range(int(window_size/2), int(16000 - window_size/2)+1, stride))

ksize = 3; psize = 2
model = keras.models.Sequential()
model.add(Conv2D(32, (ksize, ksize), activation='relu', input_shape=(n_chunks, fft_size, 1)))
model.add(MaxPooling2D(pool_size=(psize, psize)))
model.add(Conv2D(64, (ksize, ksize), activation='relu'))
model.add(MaxPooling2D(pool_size=(psize, psize)))
model.add(Conv2D(128, (ksize, ksize), activation='relu'))
model.add(MaxPooling2D(pool_size=(psize, psize)))
model.add(Conv2D(256, (ksize, ksize), activation='relu'))
model.add(MaxPooling2D(pool_size=(psize, psize)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation='softmax'))

model.compile(
    loss="categorical_crossentropy",
    optimizer='adadelta',
    metrics=["accuracy"]
)

###############################################################################
### Model fitting
model.fit_generator(
    generator=ds_train,
    steps_per_epoch=0.7 * len(df) / bsize,
    epochs=10,
    validation_data=ds_validation,
    validation_steps=0.3 * len(df) / bsize
)

###############################################################################
### Making predictions
df_validation = df[~df.index.isin(id_train)]
n_steps = len(df_validation) / bsize + 1
ds_validation = data_generator(
    df[~df.index.isin(id_train)],
    batch_size=bsize,
    shuffle=False,
    window_size_ms=win1,
    window_stride_ms=win2
)

predictions = model.predict_generator(
    generator=ds_validation,
    steps=n_steps
)
print(predictions)

classes = np.argmax(predictions, axis=1)
x = df_validation.assign(pred_word_id=classes[0:len(df_validation)])

pred_word = x['pred_word_id'].astype(str)
for i in reversed(range(len(category))):
    pred_word = pred_word.str.replace(str(i), category[i])
pred_word.name = 'pred_word'

x = pd.concat([x, pred_word], axis=1).assign(count=x.word == pred_word)
100 * (x['count'].sum() / len(x))  # Accuracy (%)

###############################################################################
### Visualization
# use the sankey.py code: https://github.com/anazalea/pySankey
import sankey
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

sankeycolors = {}
for v in range(len(category)):
    sankeycolors[category[v]] = list(colors.values())[v]

sankey.sankey(left=x['word'], right=x['pred_word'], aspect=20, colorDict=sankeycolors,
              fontsize=10, figure_name="alluvial")
