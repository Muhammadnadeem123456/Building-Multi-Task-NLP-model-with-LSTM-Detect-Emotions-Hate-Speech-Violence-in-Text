# Building-Multi-Task-NLP-model-with-LSTM-Detect-Emotions-Hate-Speech-Violence-in-Text

Multi-task learning (MTL) is a machine learning approach where a model is trained simultaneously on multiple related tasks!
**Libraries used:**

Pandas, Numpy, Scikit-learn, nltk, tensorflow, seaborn, matplotlib

Following is the architecture of the project:

**Architecture**

**Datasets:**

(i) Emotion Data: https://www.kaggle.com/datasets/nelgiriyewithana/emotions

(ii) Violence Data: https://www.kaggle.com/datasets/gauravduttakiit/gender-based-violence-tweet-classification?select=Train.csv

(iii) Hate Speech Data: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

**Compontents:**

**(i) Individual Inputs:**

Multi-Task model accepts input for each task individually.

**Model Details**
The model is a multi-input, multi-output architecture implemented using Keras and TensorFlow. It accepts the following inputs:

emotion_input: Input data for emotion classification.

violence_input: Input data for violence classification.

hate_input: Input data for hate speech classification.

The model has three output layers corresponding to the three tasks:

emotion_output: The predicted emotion class.

violence_output: The predicted violence class.

hate_output: The predicted hate class.

Model Architecture

**Inputs:**

emotion_input: A vector of features representing the emotional content.

violence_input: A vector of features representing the level of violence.

hate_input: A vector of features representing hate speech content.

**Outputs:**

emotion_output: Classification result for emotions.

violence_output: Classification result for violence.

hate_output: Classification result for hate speech.

**Loss Function:**

SparseCategoricalCrossentropy for multi-class classification tasks.

**Training**
To train the model, ensure your data is preprocessed and in the correct format. The model uses multi-task learning where each input is associated with multiple labels. The training process involves the following steps:

Prepare your data (inputs and labels).

Train the model using model.fit().

Use the provided code to train the model for 10 epochs (or adjust as necessary).

**(ii) Individual Outputs:**

Multi-Task model generates output for each task individually

**(iii) Shared Core Layers:**

Multi-Task models functionality is that a sinlge model can do multiple related tasks. To exhibit this functionality the core layers of 
the model like (Embedding, LSTM, pooling, dropout etc) are shared among all the tasks and are not seperately available for them
