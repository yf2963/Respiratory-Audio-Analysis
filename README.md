## Introduction
Lung auscultation, the process of listening to breath sounds, requires subjective decision making by clinicians. Auscultation is used as best practice for diagnosing a series of lung pathologies including pneumonia, pulmonary edema, pleural effusion and others. While traditional stethoscopes have proven effective for listening to the lungs, electronic stethoscopes and other modes of listening are becoming increasingly popular. The introduction of electronic auscultation opens the door for intelligent listening models that can assist clinicians during their diagnostic process. To assist in the detection of lung pathologies, the aim is to leverage machine learning and data analytics to detect crackles and wheezes in breathing. The detection of these features in breathing will assist in finding which patients, from a large set, need further examination for potential pathologies.

## Problem Definition
Currently, the identification of lung sounds is a subjective process that varies by clinicians. Doctors with increased experience in the field may be more accurate at identifying these sounds. This project aims to provide a sound classification model that will level the playing field among doctors and increase the specificity and sensitivity of lung auscultation. Our goal is to identify crackles and wheezes from audio files of breathing in order to build a model that is capable of aiding in diagnosing patients.

## Methods
For our methods, we'll employ two supervised algorithms CNN & RNN in order to get the most accurate results by comparing the two method's outputs. CNNs are highly efficient at extracting spatial hierarchies of features from data, which makes them ideal for processing spectrograms or MFCCs derived from lung sounds. Their ability to detect and learn patterns like crackles and wheezes from these representations is a significant advantage. RNNs, and particularly their variants like LSTM, excel in processing sequential and time-series data. They are capable of capturing the temporal dynamics and long-term dependencies in lung sounds, potentially providing a more nuanced understanding of the patterns in breathing. By comparing and contrasting the results from both models, we aim to leverage the spatial feature detection prowess of CNNs and the temporal pattern recognition capabilities of RNNs. This dual approach may provide a more comprehensive and accurate tool for clinicians in the diagnosis of lung pathologies.

### Convolutional Neural Networks (Supervised)
1. Preprocess the audio recordings to convert them into spectrograms or mel-frequency cepstral coefficients (MFCCs), suitable for CNN input.
2. Design a CNN architecture with multiple layers including convolutional layers, pooling layers, and fully connected layers.
3. Utilize convolutional layers to automatically detect and learn relevant features from the audio data, such as patterns associated with crackles and wheezes.
4. Apply pooling layers to reduce the dimensionality of the data and extract the most significant features, enhancing the network’s computational efficiency.
5. Train the CNN model using the preprocessed data and corresponding labels (types of coughs), using backpropagation and an appropriate optimizer like Adam or SGD.
6. Regularize the model to prevent overfitting by methods like dropout or data augmentation.
7. Fine-tune the network hyperparameters (like number of layers, filter sizes, learning rate) using cross-validation and experimentation to optimize performance.
8. By implementing CNNs, the model is expected to learn more complex and hierarchical feature representations from lung sounds, which could potentially lead to more accurate and reliable classification compared to KNN.

### Recurrent Neural Networks (Supervised)
1. Preprocess the audio data to ensure it is in a suitable format for RNN processing, often involving normalization and possibly converting to a time-series format.
2. Utilize a Recurrent Neural Network architecture, which is especially well-suited for sequential data like audio, to capture the temporal dynamics of lung sounds.
3. Employ layers such as LSTM (Long Short-Term Memory) to effectively handle long-term dependencies and learn patterns over different time scales in the audio data.
4. Incorporate dropout layers or use techniques like recurrent dropout to prevent overfitting, ensuring the model generalizes well to new, unseen data.
5. Train the RNN using the preprocessed lung sound data, employing backpropagation through time (BPTT) and an appropriate optimizer like NADAM.
6. Regularly validate the model's performance on a separate validation dataset to monitor for overfitting and to tune hyperparameters.
7. Experiment with different RNN architectures and the number of layers to find the best model structure for accurately identifying crackles and wheezes in the lung sounds.
8. Finally, evaluate the trained RNN model on a test set and compare its performance with the CNN model to determine the most effective approach for lung sound classification.

### Mel-Spectrogram
1. Determine repsiratory cycle and get otimal splitting length for sound files
2. Load audio files and create dataframe
3. Use librosa to apply to data and create spectrogram
4. Convert this spectrogram to decibels
5. Plot the spectrogram

The mel spectrogram for preprocessing has the advantage of being able to mimic human hearing perceptions by adjusting relative frequency differences to be similiar to how people would interpret the sound. Additionally, it is able to reduce the dimensionality of the training data and get rid of noise through binning and mel bands process. Because it is a logarithmic scale frequencies that would cause noise in the original recording would be filtered out more after the mel spectrogram because these frequencies would not be interpreted by humans.

### Mel-Frequency Cepstral Coefficients (MFCC)
1. Begin by segmenting the audio signal into short frames, as human speech varies rapidly over time.
2. For each frame, compute the spectrum using the Fourier Transform to convert from time domain to frequency domain.
3. Apply the Mel scale to the power spectrum to account for human auditory perception, which does not perceive pitches linearly.
4. Take the logarithm of the Mel spectrum to stabilize the variance.
5. Finally, compute the Discrete Cosine Transform (DCT) of the log Mel powers to obtain the MFCCs.

MFCCs are especially effective for sound classification tasks as they succinctly represent the power spectrum of a sound. By focusing on the human auditory system's perception of sound, MFCCs are more robust to noise and variations in the audio signal, making them ideal for modeling complex sounds like lung auscultations.

### Short-Time Fourier Transform (STFT)
1. Divide the audio signal into short, overlapping segments or frames to capture the time-varying properties of the signal.
2. Apply window function to each frame to minimize the signal discontinuities at the beginning and end of each frame.
3. Compute the Fourier Transform for each frame. This transformation converts the time domain signal in each frame into a frequency domain representation.
4. Obtain the magnitude and phase of the Fourier coefficients, which represent the amplitude and phase information of the signal at various frequencies.
5. Visualize the STFT as a spectrogram, where the x-axis represents time, the y-axis represents frequency, and the intensity of color indicates the amplitude of a particular frequency at a given time.

The STFT is valuable for its ability to analyze both stationary and non-stationary signals, capturing the frequency and phase information over time. It is widely used in signal processing for time-frequency analysis, especially in contexts where the frequency content of a signal varies over time, such as in the analysis of lung sounds.

### Combined Dense Feature: MFCC, STFT, and Mel Spectrogram
1. Feature Extraction: Begin by individually extracting the three features from the audio recordings. This involves computing the MFCCs, generating the STFT, and creating the Mel Spectrogram for each audio sample.
2. Feature Concatenation: After extraction, concatenate these features to form a comprehensive feature set. Each feature contributes unique information: MFCCs capture the melodic aspect, STFT provides time-frequency analysis, and the Mel Spectrogram offers a perception-aligned frequency representation.
3. Dimensionality Reduction: Due to the high dimensionality of the combined feature set, apply techniques like Principal Component Analysis (PCA) or Autoencoders to reduce the dimensionality. This step ensures computational efficiency and mitigates the risk of overfitting.
4. Normalization and Scaling: Normalize the combined feature set to ensure that all features contribute equally to the model's learning process. This step is crucial for maintaining the balance between the features.
5. Dense Feature Representation: The concatenated and processed features form a dense representation of the audio signal. This dense feature encapsulates a comprehensive understanding of the signal, leveraging the strengths of each individual feature.
6. Input to Machine Learning Models: Use this dense feature as input to your CNN and SVM models. The rich information embedded in the combined feature set is expected to enhance the models' ability to discern subtle patterns in lung sounds, thereby improving the accuracy and reliability of pathology detection.
7. Model Training and Evaluation: Train your models using this combined feature set and evaluate their performance. The synergy of MFCC, STFT, and Mel Spectrogram within the dense feature set should ideally lead to more nuanced and robust classifications.

The integration of MFCC, STFT, and Mel Spectrogram into a single dense feature set is a powerful approach in audio signal processing. This method harnesses the individual strengths of each feature type, potentially leading to more accurate and effective sound classification in the context of lung auscultation and pathology detection.

## Data Collection
It's important to establish reliable metrics for evaluating the effectiveness of CNN and RNN models in classifying lung pathologies. Given that most respiratory patterns are within the normal range, and considering the differing implications of false positives and negatives, the F1 score emerges as  metric for this task. This metric is particularly valuable when one classification category is more prevalent than others, and when there are distinct costs associated with false positives and false negatives.

During the training phase, the loss function employed was sparse categorical cross-entropy. This choice is good for multi-class categorization scenarios as it evaluates the accuracy of the model by comparing the actual categorization (ground truth) against the predicted probability distribution across all classes. This loss function is important in fine-tuning the models, especially in complex tasks like lung pathology classification, where the distinction between different categories is subtle yet critical for accurate diagnosis.

## Results and Discussion

## Step 1: Data Importing
This step creates a dataframe with start, end, (cycle start end) crackles, wheezes, pid, and filename

![image](https://github.com/user-attachments/assets/5a3f4ede-7b09-4b72-9bcc-014a38d19eed)

## Step 2: Data Preprocessing
This step splits the audio into cycles and updates the dataframe with the new filenames. In order to split up audio, we must choose a standard length for all audio samples, however a sample too large would be difficult to process and a sample too small may eliminate too much data. We chose 6 seconds as a standard length for all audio samples (see figure).

![image](https://github.com/user-attachments/assets/52a26549-e271-41f4-aac3-60ade46168a8)

## Step 3: Data Categorization and Division
This step adds categories to each cycle based on the presence of crackles and wheezes. The categories are as follows: none, crackles only, wheezes only, and both crackles and wheezes. The categories are added to the dataframe as a new column. Furthermore, the data is divided into training and testing sets based on each category. As seen in the "Category Distribution" figure, the dataset is imbalanced.

![image](https://github.com/user-attachments/assets/58bb9437-e899-4dba-ae49-89b55b9045aa)

## Step 4: Feature Extraction and Modeling
This creates CNN and RNN models for three different features, MFCC, STFT, and Mel-Spectrogram which are all ways of compressing audio information. Furthermore, this step creates a dense network, combining all three features into a singular CNN and RNN models.

### Step 4.1: Mel-Frequency Cepstral Coefficients (MFCC)
![image](https://github.com/user-attachments/assets/3d3eaf45-f749-40bd-a80c-e7ab0bb785ae)

### Step 4.2: Short-Time Fourier Transform (STFT)
![image](https://github.com/user-attachments/assets/fcb926fc-7139-4eb0-8053-e8a394867e48)

### Step 4.3: Mel-Spectrogram
![image](https://github.com/user-attachments/assets/a1f45d2f-5adb-4a42-8603-dbb478602ca6)

## Step 5: Data Training
This step uses the models created in step 4 and trains them on the training data. The training data is then tested on the testing data and the results are plotted. Figures showing the accuracy and loss of each model are shown below.

### Step 5.1: MFCC
#### Step 5.1.1: MFCC CNN
![image](https://github.com/user-attachments/assets/1612c521-b8b9-4c3d-a95b-d44732941b8b)

#### Step 5.1.2: MFCC RNN
![image](https://github.com/user-attachments/assets/5e7f87d4-998c-4876-a4bc-bcbda143ab69)

### Step 5.2: STFT
#### Step 5.2.1: STFT CNN
![image](https://github.com/user-attachments/assets/b53d200a-6c4d-4fad-ac6b-f25d91ced6cb)

#### Step 5.2.2: STFT RNN
![image](https://github.com/user-attachments/assets/a2b3cdde-6d34-4aec-8256-87d104e64631)

### Step 5.3: Mel
#### Step 5.3.1: Mel-Spectrogram CNN
![image](https://github.com/user-attachments/assets/02bc9cc1-d5ff-4c4f-9377-ec1a0f8b3623)

#### Step 5.3.2: Mel-Spectrogram RNN
![image](https://github.com/user-attachments/assets/69025951-e903-477a-8953-218d32a6eac0)

### Step 5.4: Dense
#### Step 5.4.1: Dense CNN
![image](https://github.com/user-attachments/assets/9ce93ee6-dbec-4ad2-a382-d4b44c69d5d7)

![image](https://github.com/user-attachments/assets/e3c84dcc-7b3b-4726-95ca-541cbeb08697)

![image](https://github.com/user-attachments/assets/d96b6a7c-c6ad-4cd1-8f56-8670499bf09b)

#### Step 5.4.2: Dense RNN
![image](https://github.com/user-attachments/assets/6b5e4986-bc04-420d-97f8-dcd533531fd0)

![image](https://github.com/user-attachments/assets/bb7296cc-e505-4cbc-89f7-efa422392309)

![image](https://github.com/user-attachments/assets/4b75b65c-467b-457c-9747-7e2d2ab3f8ac)

#### Step 5.4.3: Dense Combined
![image](https://github.com/user-attachments/assets/6e9f6dea-d56a-4b15-9966-0051c24d1e5a)

![image](https://github.com/user-attachments/assets/6fd79b5e-ab87-4d43-843d-976062476e78)

![image](https://github.com/user-attachments/assets/00d44e34-18ca-4afe-8ccc-ae97f705368a)

## Conclusion
When running each feature individually both RNN and CNN perform similarly between around 50-60% accuracy. When combining all three features and running Dense CNN and Dense RNN, we see quite different results. CNN performs significantly better at 68% accuracy whereas RNN doesn't see any improvement at 56%. Taking a look at the classification metrics of Dense RNN, we see an empty 'Both' class. This is due to overfitting of our features as we don't have enough audio samples where both crackles and wheezes are present which explains the difference in loss incurred by RNN when comparing it to Dense CNN and Dense Combined.

Our attempt at resolving this issue by simple oversampling the data to have equal proportions of all four categories skewed the results more significantly than adhering to the real world distribution. This is also the reason RNN isn't necessarily a great fit for our use case, and why CNN is the most commonly used method for audio classification models. Dense Combined, which is both CNN and RNN using all three features, scores the highest at 70% accuracy, just slightly above CNN. It's important to note that on average when training the data, Dense Combined did normally perform worse than Dense CNN. However, as training the model takes a significant amount of time given the size of the data, we were unable to get an average of these values and this is the most recent iteration our of testing and debugging. Even though Dense Combined scored the highest in accuracy, we can confidently conclude that the best method is Dense CNN. Given Dense RNN's low accuracy and high loss, we'd likely be able to get the best results from tweaking hyperparameters and other factors using CNN, as RNN would likely only contribute to higher loss values.

## Ways to Improve the Model

1. **Synthetic Overfitting for Unbalanced Data**: We previously attempted simple oversampling to balance our dataset, which did not yield the desired results. A better approach would involve synthetic overfitting, using techniques like SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples in the dataset. This method could help in creating a more balanced dataset, thereby improving the model's ability to learn from under-represented classes without the distortion that often accompanies simple oversampling.

2. **Hyperparameter Tuning**: Optimizing hyperparameters is another important area for improvement. By systematically experimenting with various combinations of learning rates, dropout rates, and number of layers, we can fine-tune our models to achieve better performance. Hyperparameter tuning can be especially impactful in deep learning models like CNNs, where small adjustments can lead to significant changes in model efficacy.

3. **Enhanced Data Cleaning and Preprocessing**: Further refining our data cleaning and preprocessing steps can also contribute to model improvement. Techniques such as data smoothing can help in reducing noise and making patterns in the data more discernible for the models. This step is particularly important in the context of lung pathology classification, where subtle differences in audio patterns can be indicative of different conditions.
   
## References

1. Gurung, A., Scrafford, C. G., Tielsch, J. M., Levine, O. S., & Checkley, W. (2011). Computerized lung sound analysis as diagnostic aid for the detection of abnormal lung sounds: A systematic review and meta-analysis. *Respiratory Medicine*, *105*(9), 1396–1403. https://doi.org/10.1016/j.rmed.2011.05.007

2. Rao, A., Huynh, E., Royston, T. J., Kornblith, A., & Roy, S. (2019). Acoustic Methods for Pulmonary Diagnosis. *IEEE Reviews in Biomedical Engineering*, *12*, 221–239. https://doi.org/10.1109/RBME.2018.2874353

3. Palaniappan, R., Sundaraj, K., & Sundaraj, S. (2014). A comparative study of the svm and k-nn machine learning algorithms for the diagnosis of respiratory pathologies using pulmonary acoustic signals. *BMC Bioinformatics*, *15*(1), 223–223. https://doi.org/10.1186/1471-2105-15-223

