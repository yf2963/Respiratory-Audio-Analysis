import os

import numpy as np
import pandas as pd
import librosa as lb
import soundfile as sf
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.layers as lay
import tensorflow.keras.callbacks as call
import librosa.display as ld


def ensureDirectoryExists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def getFiles(path='../archive/Respiratory_Sound_Database/audio_and_txt_files/'):
    files = []
    for file in os.listdir(path):
        if ('.txt' in file):
            files.append(file.split('.')[0])
    return files


def getCompleteData(path='../archive/Respiratory_Sound_Database/audio_and_txt_files/'):
    data = [
        pd.read_csv(os.path.join(path, f"{file}.txt"), sep='\t', names=['start', 'end', 'crackles', 'wheezes'])
        .assign(pid=file.split('_')[0], filename=file)
        for file in getFiles(path)
    ]
    allDataUnsplit = pd.concat(data, ignore_index=True)
    ensureDirectoryExists('data')
    allDataUnsplit.to_csv('data/unsplit.csv', index=False)
    return allDataUnsplit


def loadAudio(path, start, end, maxLength):
    audio, sample = lb.load(path)
    segment = audio[int(start * sample): int(end * sample)]
    padded = lb.util.pad_center(segment, maxLength * sample)
    return padded, sample


def processAudio(data, path, maxLength=6):
    ensureDirectoryExists('data/audioSplit')
    processed = []
    i = 0
    for index, row in data.iterrows():
        start, end = row['start'], row['end']
        if end - start > maxLength:
            end = start + maxLength

        if index > 0:
            if data.iloc[index - 1]['filename'] == row['filename']:
                i += 1
            else:
                i = 0

        audioPath = os.path.join(path, f"{row['filename']}.wav")
        padded, sample = loadAudio(audioPath, start, end, maxLength)

        newFilename = f"{row['filename']}_{i}.wav"
        sf.write(file=os.path.join('data/audioSplit', newFilename), data=padded, samplerate=sample)

        row['filename'] = newFilename
        processed.append(row)

    return pd.DataFrame(processed)


def getSplitData(unsplitData, path='../archive/Respiratory_Sound_Database/audio_and_txt_files/', maxLength=6):
    ensureDirectoryExists('data')
    dataSplitPath = 'data/split.csv'

    if os.path.exists(dataSplitPath) and os.path.exists('data/audioSplit'):
        return pd.read_csv(dataSplitPath)
    else:
        processed = processAudio(unsplitData, path, maxLength)
        processed.to_csv(dataSplitPath, index=False)
        return processed


def getCycleGraph(data, maxLength=6):
    cycleList = [row['end'] - row['start'] for index, row in data.iterrows()]
    cycleArray = np.array(cycleList)
    plt.style.use('ggplot')
    plt.hist(cycleArray, bins=60, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(maxLength, color='red', linestyle='dashed', linewidth=2)
    plt.legend(['Max Length Threshold', 'Cycle Lengths'])
    plt.title('Cycle Length Distribution')
    plt.xlabel('Cycle Length (seconds)')
    plt.ylabel('Number of Samples')
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    under = (np.sum(cycleArray < maxLength) / len(cycleArray)) * 100
    print(f'{under:.2f} percent of cycles are less than {maxLength} seconds long')


def getCategorizedData(data):
    conditions = [
        (data['crackles'] == 0) & (data['wheezes'] == 0),
        (data['crackles'] == 1) & (data['wheezes'] == 0),
        (data['crackles'] == 0) & (data['wheezes'] == 1),
        (data['crackles'] == 1) & (data['wheezes'] == 1),
    ]
    choices = ['none', 'crackles', 'wheezes', 'both']
    newData = data.copy()
    newData['category'] = np.select(conditions, choices, default=np.nan)
    getCountPlot(newData)
    return newData


def getCountPlot(data, x='category', title='Category Distribution'):
    plt.figure(figsize=(10, 6))  # Set the figure size
    ax = sns.countplot(x=x, data=data)  # Create the count plot
    ax.set_title(title)  # Set the title of the plot
    ax.set_xlabel('Category')  # Set the x-axis label
    ax.set_ylabel('Count')  # Set the y-axis label
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # Display the counts above the bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 10),
                    textcoords='offset points')
    plt.tight_layout()  # Adjust the plot to ensure everything fits without overlap
    plt.show()  # Display the plot


def getTestTrainSplit(data, testSize=0.20, randomstate=69):
    dataTrain, dataTest, _, _ = train_test_split(data, data.category, stratify=data.category, random_state=69,
                                                 test_size=testSize)
    print('Training data category distribution:')
    print(dataTrain.category.value_counts() / dataTrain.shape[0])
    print(f'total: {dataTrain.shape[0]}')
    print('\n')
    print('Test data category distribution:')
    print(dataTest.category.value_counts() / dataTest.shape[0])
    print(f'total: {dataTest.shape[0]}')
    return dataTrain, dataTest


def encode(trainData, testData):
    le = LabelEncoder()
    return le.fit_transform(trainData), le.transform(testData)


def getShape(data):
    return data.shape[1:] + (1,)


def getFeatureData(data, feature, path='data/audioSplit/'):
    features = []
    for index, row in data.iterrows():
        audioPath = os.path.join(path, row['filename'])
        audio, sample = lb.load(audioPath)
        if (feature == 'mfcc'):
            feat = lb.feature.mfcc(y=audio, sr=sample)
        elif (feature == 'stft'):
            feat = lb.feature.chroma_stft(y=audio, sr=sample)
        elif (feature == 'mel'):
            feat = lb.feature.melspectrogram(y=audio, sr=sample)
        features.append(feat)
    return np.array(features)


def getMFCCModel(trainData, testData):
    mfccTrain = getFeatureData(trainData, feature='mfcc')
    mfccTest = getFeatureData(testData, feature='mfcc')
    mfccShape = getShape(mfccTrain)
    mfccCnnIn = lay.Input(shape=mfccShape, name='mfccCnnIn')
    k = lay.Conv2D(32, 5, strides=(1, 3), padding='same')(mfccCnnIn)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    k = lay.MaxPooling2D(pool_size=2, padding='valid')(k)
    k = lay.Conv2D(64, 3, strides=(1, 2), padding='same')(k)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    k = lay.MaxPooling2D(pool_size=2, padding='valid')(k)
    k = lay.Conv2D(96, 2, padding='same')(k)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    k = lay.MaxPooling2D(pool_size=2, padding='valid')(k)
    k = lay.Conv2D(128, 2, padding='same')(k)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    mfccCnnOut = lay.GlobalMaxPooling2D()(k)
    mfccCnnModel = ks.Model(mfccCnnIn, mfccCnnOut, name="mfccCnnModel")

    mfccRnnIn = lay.Input(shape=(mfccShape[:-1]), name='mfccRnnIn')
    x = lay.LSTM(units=128, return_sequences=True)(mfccRnnIn)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.LSTM(units=128, return_sequences=True)(x)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.LSTM(units=128)(x)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.Dense(units=128 // 2, activation='relu')(x)
    x = lay.Dropout(0.3)(x)
    mfccRnnOut = lay.Dense(units=4, activation='softmax')(x)
    mfccRnnModel = ks.Model(inputs=mfccRnnIn, outputs=mfccRnnOut, name='mfccRnnModel')

    return mfccCnnModel, mfccRnnModel, mfccTrain, mfccTest


def getSTFTModel(trainData, testData):
    stftTrain = getFeatureData(trainData, feature='stft')
    stftTest = getFeatureData(testData, feature='stft')
    stftShape = getShape(stftTrain)

    stftCnnIn = lay.Input(shape=stftShape, name='stftCnnIn')
    k = lay.Conv2D(32, 5, strides=(1, 3), padding='same')(stftCnnIn)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    k = lay.MaxPooling2D(pool_size=2, padding='valid')(k)
    k = lay.Conv2D(64, 3, strides=(1, 2), padding='same')(k)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    k = lay.MaxPooling2D(pool_size=2, padding='valid')(k)
    k = lay.Conv2D(128, 2, padding='same')(k)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    stftCnnOut = lay.GlobalMaxPooling2D()(k)
    stftCnnModel = ks.Model(stftCnnIn, stftCnnOut, name="stftCnnModel")

    stftRnnIn = lay.Input(shape=(stftShape[:-1]), name='stftRnnIn')
    x = lay.LSTM(units=128, return_sequences=True)(stftRnnIn)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.LSTM(units=128, return_sequences=True)(x)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.LSTM(units=128)(x)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.Dense(units=128 // 2, activation='relu')(x)
    x = lay.Dropout(0.3)(x)
    stftRnnOut = lay.Dense(units=4, activation='softmax')(x)
    stftRnnModel = ks.Model(inputs=stftRnnIn, outputs=stftRnnOut, name='stftRnnModel')

    return stftCnnModel, stftRnnModel, stftTrain, stftTest


def getMELModel(trainData, testData):
    melTrain = getFeatureData(trainData, feature='mel')
    melTest = getFeatureData(testData, feature='mel')
    melShape = getShape(melTrain)

    melCnnIn = lay.Input(shape=melShape, name='melCnnIn')
    k = lay.Conv2D(32, 5, strides=(2, 3), padding='same')(melCnnIn)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    k = lay.MaxPooling2D(pool_size=2, padding='valid')(k)
    k = lay.Conv2D(64, 3, strides=(2, 2), padding='same')(k)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    k = lay.MaxPooling2D(pool_size=2, padding='valid')(k)
    k = lay.Conv2D(96, 2, padding='same')(k)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    k = lay.MaxPooling2D(pool_size=2, padding='valid')(k)
    k = lay.Conv2D(128, 2, padding='same')(k)
    k = lay.BatchNormalization()(k)
    k = lay.Activation(ks.activations.relu)(k)
    melCnnOut = lay.GlobalMaxPooling2D()(k)
    melCnnModel = ks.Model(melCnnIn, melCnnOut, name="melCnnModel")

    melRnnIn = lay.Input(shape=(melShape[:-1]), name='melRnnIn')
    x = lay.LSTM(units=128, return_sequences=True)(melRnnIn)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.LSTM(units=128, return_sequences=True)(x)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.LSTM(units=128)(x)
    x = lay.Dropout(0.3)(x)
    x = lay.BatchNormalization()(x)
    x = lay.Dense(units=128 // 2, activation='relu')(x)
    x = lay.Dropout(0.3)(x)
    melRnnOut = lay.Dense(units=4, activation='softmax')(x)
    melRnnModel = ks.Model(inputs=melRnnIn, outputs=melRnnOut, name='melRnnModel')

    return melCnnModel, melRnnModel, melTrain, melTest


def getDenseModel(mfccModel, stftModel, melModel):
    mfccIn = lay.Input(shape=mfccModel.input_shape[1:], name="mfcc")
    mfcc = mfccModel(mfccIn)
    stftIn = lay.Input(shape=stftModel.input_shape[1:], name="stft")
    stft = stftModel(stftIn)
    melIn = lay.Input(shape=melModel.input_shape[1:], name="mel")
    mel = melModel(melIn)

    j = lay.concatenate([mfcc, stft, mel])
    k = lay.Dropout(0.2)(j)
    k = lay.Dense(50, activation='relu')(j)
    k = lay.Dropout(0.3)(k)
    k = lay.Dense(25, activation='relu')(k)
    k = lay.Dropout(0.3)(k)
    output = lay.Dense(8, activation='softmax')(k)

    model = ks.Model(inputs=[mfccIn, stftIn, melIn], outputs=output, name="DenseModel")
    return model


def getDenseModel2(mfccCnnModel, stftCnnModel, melCnnModel, mfccRnnModel, stftRnnModel, melRnnModel):
    mfccCnnIn = lay.Input(shape=mfccCnnModel.input_shape[1:], name="mfccCnnIn")
    mfccCnn = mfccCnnModel(mfccCnnIn)
    stftCnnIn = lay.Input(shape=stftCnnModel.input_shape[1:], name="stftCnnIn")
    stftCnn = stftCnnModel(stftCnnIn)
    melCnnIn = lay.Input(shape=melCnnModel.input_shape[1:], name="melCnnIn")
    melCnn = melCnnModel(melCnnIn)
    mfccRnnIn = lay.Input(shape=mfccRnnModel.input_shape[1:], name="mfccRnnIn")
    mfccRnn = mfccRnnModel(mfccRnnIn)
    stftRnnIn = lay.Input(shape=stftRnnModel.input_shape[1:], name="stftRnnIn")
    stftRnn = stftRnnModel(stftRnnIn)
    melRnnIn = lay.Input(shape=melRnnModel.input_shape[1:], name="melRnnIn")
    melRnn = melRnnModel(melRnnIn)

    j = lay.concatenate([mfccCnn, stftCnn, melCnn, mfccRnn, stftRnn, melRnn])
    k = lay.Dropout(0.2)(j)
    k = lay.Dense(50, activation='relu')(j)
    k = lay.Dropout(0.3)(k)
    k = lay.Dense(25, activation='relu')(k)
    k = lay.Dropout(0.3)(k)
    output = lay.Dense(8, activation='softmax')(k)

    model = ks.Model(inputs=[mfccCnnIn, stftCnnIn, melCnnIn, mfccRnnIn, stftRnnIn, melRnnIn], outputs=output,
                     name="DenseCombinedModel")
    return model


def trainModel(model, dtr, dte, input, valid, learningRate=0.001, epochs=100, batchSize=32):
    model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])
    ks.backend.set_value(model.optimizer.learning_rate, learningRate)
    history = model.fit(
        input,
        dtr,
        validation_data=(valid, dte),
        epochs=epochs,
        batch_size=batchSize,
        verbose=1,
        callbacks=[
            call.EarlyStopping(patience=5),
            call.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001, mode='min')
        ]
    )
    return model, history


def plotModel(history):
    historyDf = pd.DataFrame(history.history)
    ymaxAcc = max(1, historyDf['accuracy'].max(), historyDf['val_accuracy'].max())
    ymaxLoss = max(1, historyDf['loss'].max(), historyDf['val_loss'].max())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    historyDf[['accuracy', 'val_accuracy']].plot(ax=ax1)
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, ymaxAcc])
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax1.grid(True)
    historyDf[['loss', 'val_loss']].plot(ax=ax2)
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_ylim([0, ymaxLoss])
    ax2.legend(['Train', 'Validation'], loc='upper right')
    ax2.grid(True)
    plt.tight_layout()
    plt.show()


def createMFCCPlot(mfccTrain, index=420):
    feature = mfccTrain[index]
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ld.specshow(lb.power_to_db(feature, ref=np.max), x_axis='time', ax=ax, cmap='viridis')
    colorbar = fig.colorbar(img, ax=ax)
    colorbar.set_label('Decibels (dB)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MFCC Coefficients')
    ax.set_title('MFCC')
    plt.tight_layout()
    plt.show()


def createSTFTPlot(stftTrain, index=420):
    feature = stftTrain[index]
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ld.specshow(lb.amplitude_to_db(feature), x_axis='time', y_axis='linear', ax=ax, cmap='viridis')
    colorbar = fig.colorbar(img, ax=ax)
    colorbar.set_label('Magnitude (dB)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('STFT Spectrogram')
    plt.tight_layout()
    plt.show()


def createMELPlot(melTrain, index=420):
    feature = melTrain[index]
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ld.specshow(lb.power_to_db(feature, ref=np.max), x_axis='time', y_axis='mel', ax=ax, cmap='viridis')
    colorbar = fig.colorbar(img, ax=ax)
    colorbar.set_label('Amplitude (dB)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency')
    ax.set_title('Mel-Spectrogram')
    ax.set(title='Mel-Spectrogram')
    plt.show()


def plotModelLayers(model, filename, show_shapes=True):
    ks.utils.plot_model(model, to_file=filename, show_shapes=show_shapes, show_layer_names=True)


# none:2, both: 0, crackles: 1, wheezes: 3

def plotClassificationAndConfusion(model, xt, yt):
    predictions = model.predict(xt)
    predicted_categories = np.argmax(predictions, axis=1)
    conf_matrix = confusion_matrix(yt, predicted_categories)
    print("Confusion Matrix:\n", conf_matrix)
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.xaxis.set_ticklabels(["Both", "Crackles", "None", "Wheezes"])
    ax.yaxis.set_ticklabels(["Both", "Crackles", "None", "Wheezes"])
    plt.show()

    class_report = classification_report(yt, predicted_categories)
    print("Classification Report:\n", class_report)
    report = classification_report(yt, predicted_categories, output_dict=True,
                                   target_names=["Both", "Crackles", "None", "Wheezes"])
    report_df = pd.DataFrame(report).transpose()
    report_df.drop(columns=['support'], inplace=True, errors='ignore')
    report_df[:-3].plot(kind='bar', figsize=(10, 6))
    plt.title("Classification Metrics per Class")
    plt.xlabel("Classes")
    plt.ylabel("Scores")
    plt.show()
