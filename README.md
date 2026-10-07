# 🧠 Deep Learning IMDB Sentiment Analysis using SimpleRNN

A Deep Learning based Natural Language Processing project that uses an Embedding layer and SimpleRNN to classify IMDB movie reviews as Positive or Negative.

## Project Overview

This project explores **sentiment analysis**: using text to estimate whether an opinion is positive or negative. That kind of classification can help organize or summarize large collections of reviews.

The project uses the **IMDB Movie Reviews** dataset provided by Keras and demonstrates an end-to-end Deep Learning NLP workflow. Reviews are converted from words to integer sequences, padded to a fixed length, and passed into a trained Embedding + SimpleRNN model. A Streamlit web app loads the saved model and lets a user analyze review text.

The model output is a prediction score, not a guarantee that the review has been interpreted correctly.

## What I Learned From This Project

| Concept | What it means in this project |
|---|---|
| Natural Language Processing | Preparing review text so a neural network can process it. |
| Text preprocessing | Lowercasing text, splitting it on whitespace, mapping words to IDs, and padding sequences. |
| Tokenization and word indexing | Representing words with integer IDs from the Keras IMDB word index. |
| Vocabulary | The trained model's Embedding layer accepts IDs in a vocabulary of 10,000 entries. |
| Integer encoding | Replacing each input word with a numeric ID. |
| Padding sequences | Making each model input exactly 500 IDs long. |
| Embedding layer | Learning a 128-dimensional representation for each vocabulary ID. |
| SimpleRNN | Processing the sequence in order with 128 recurrent units. |
| Activation functions | Using `tanh` in the recurrent layer and `sigmoid` in the output layer. |
| Binary classification | Choosing one of two labels: Positive or Negative. |
| Binary cross-entropy | Measuring prediction error for the two-class task during training. |
| Adam optimizer | Updating model weights during training to reduce loss. |
| Forward propagation | Passing a padded sequence through the model to produce a score. |
| Model training and validation | Training in batches and using a validation split to monitor `val_loss`. |
| Prediction | Applying the saved model to a user-entered review. |
| Model saving and loading | Saving the trained network as an HDF5 `.h5` file and loading it in the app. |
| Streamlit application | Providing an interactive interface for review analysis. |

## Dataset

The project uses the **IMDB Movie Reviews** dataset available through `tensorflow.keras.datasets.imdb`. It is used for binary sentiment classification, with positive and negative labels.

The training notebook loads the dataset's train and test splits with `num_words=10000`. Reviews in this dataset are supplied as **integer word IDs**, not raw text. The notebook pads the sequences to a maximum length of 500 before training. During training, 20% of the training data is used as validation data.

```text
Review text
    ↓
Word IDs
    ↓
Padding to 500 IDs
    ↓
Embedding
    ↓
SimpleRNN
    ↓
Sigmoid score
    ↓
Positive / Negative
```

## How Text Is Converted Into Numbers

Keras provides an IMDB `word_index`, a mapping from words to integer ranks. For example, a simplified conceptual review might be represented like this:

```text
"This movie is amazing"
        ↓
[integer IDs from the IMDB word index]
```

The exact IDs depend on the IMDB word index; the list above is illustrative, not a literal project output.

- **`word_index`** maps words to their IMDB integer ranks.
- **`reverse_word_index`** maps ranks back to words. The notebooks use it to inspect or decode a dataset review; it is not needed for the Streamlit prediction path.
- **Reserved IDs and offset:** the dataset convention reserves IDs for padding/start/unknown tokens. Known word ranks are shifted by `+3`, and notebook decoding subtracts `3` before looking a word up.
- **Unknown words:** the app's current expression is `word_index.get(word, 2) + 3`. As written, a word absent from the index therefore becomes ID `5`. This differs from the OOV ID `2` used by Keras's `imdb.load_data` encoding.
- **Padding:** shorter reviews are padded with zeros and longer reviews are truncated to the configured fixed length.

Neural networks operate on numerical tensors rather than raw strings. Word indexing and padding turn variable-length text into a fixed-shape numeric input that the model can process.

## Preprocessing Pipeline

The Streamlit app's current prediction preprocessing follows these steps:

1. Trim whitespace at the beginning and end of the review.
2. Convert the review to lowercase.
3. Split on whitespace using Python's `str.split()`.
4. Look up each resulting token in the IMDB `word_index`, then add the IMDB offset of `3`.
5. For a missing token, use the current fallback value of `2` before adding the offset.
6. Use Keras `pad_sequences` to pad or truncate the review to 500 IDs.
7. Pass the padded sequence to the saved model.

The core implementation is:

```python
words = text.lower().split()
encoded_review = [word_index.get(word, 2) + 3 for word in words]
padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
```

This documents the existing inference code; it does not apply additional token normalization or vocabulary filtering.

## Model Architecture

The architecture below is confirmed by the saved model and the training notebook:

```text
Input review (500 integer IDs)
          ↓
Embedding (10,000 vocabulary entries, 128 dimensions)
          ↓
SimpleRNN (128 units, tanh)
          ↓
Dense (1 unit, sigmoid)
          ↓
Positive-class score
          ↓
Positive if score > 0.5; otherwise Negative
```

| Layer | Confirmed configuration | Purpose |
|---|---|---|
| Input | 500 integer IDs per review | Receives a padded sequence. |
| Embedding | Input dimension 10,000; output dimension 128 | Converts each word ID into a learned dense vector. |
| SimpleRNN | 128 units; `tanh` activation | Processes the sequence in order and learns from its context. |
| Dense | 1 unit; `sigmoid` activation | Produces a scalar score for binary classification. |

The sigmoid activation is part of the final Dense layer; it is not a separate model layer.

## Why Embedding Is Used

A word ID is only a label; its numeric size does not express the word's meaning. A one-hot vector would represent a word with a mostly-zero vector as large as the vocabulary. An **Embedding** instead looks up a compact, learned vector for each ID:

```text
Word ID → Embedding lookup → 128-value dense vector
```

During training, words used in similar contexts can learn similar vector representations. The project uses an embedding dimension of 128, as confirmed in the saved model.

## Why SimpleRNN Is Used

A Recurrent Neural Network (RNN) reads a sequence one step at a time. In a movie review, word order and context matter: for example, negation can change the meaning of a phrase. The RNN's hidden state carries information from earlier words as later words are processed.

Conceptually, a recurrent step can be written as:

```text
h_t = f(W_x x_t + W_h h_(t-1) + b)
```

- `x_t` is the current word's embedding.
- `h_(t-1)` is the state from the previous step.
- `W_x` and `W_h` are learned weights.
- `b` is a learned bias.
- `f` is the recurrent activation function, `tanh` in this model.

## Activation Functions

### `tanh`

The SimpleRNN uses `tanh` to transform its recurrent values. Its output is bounded between **-1 and 1**, allowing the recurrent layer to represent positive and negative activations.

### `sigmoid`

The final Dense layer uses `sigmoid`, which maps its output to the range **0 to 1**. The app treats this as the positive-class score:

- A score **greater than 0.5** is labeled **Positive**.
- A score **less than or equal to 0.5** is labeled **Negative**.

The app's confidence display is derived from the model score. It describes model confidence, not factual certainty or proven reliability.

## Loss Function

The model is compiled with **binary cross-entropy**, a loss function for comparing a binary target with a predicted probability:

```text
L = -[y log(p) + (1-y) log(1-p)]
```

- `y` is the actual label (0 or 1).
- `p` is the model's predicted probability for the positive class.

The loss penalizes predictions that differ from the training labels. Binary cross-entropy is appropriate here because the task has two classes and the model outputs a sigmoid score.

## Optimizer

The saved model and training notebook use the **Adam** optimizer. An optimizer updates a model's learned weights to reduce the loss. Adam adapts its weight updates using information from recent gradients, which makes it a commonly used general-purpose choice for neural network training.

## Training Process

The training notebook uses this general training loop:

```text
Training batch
     ↓
Forward pass
     ↓
Prediction
     ↓
Loss calculation
     ↓
Backpropagation and gradient calculation
     ↓
Adam weight update
     ↓
Next batch / epoch
```

The notebook specifies a maximum of **20 epochs**, a **batch size of 32**, and `validation_split=0.2`. It also configures EarlyStopping to monitor `val_loss`, stop after 5 epochs without improvement, and restore the best weights. Because EarlyStopping may stop training early, 20 is a maximum rather than a claim about the number of epochs completed.

- An **epoch** is one pass over the training portion of the data.
- A **batch** is the subset processed before a weight update.
- **Loss** measures prediction error according to the chosen loss function.
- **Accuracy** is the fraction of predictions that match their labels; the notebook requests it as a training metric.
- **Validation data** is held out from weight updates and used to monitor generalization during training. Here it is selected using the 20% validation split.

No accuracy or other evaluation metric is reported here because the repository does not provide a verified evaluation result to cite.

## Streamlit Application

The root `main.py` file implements the Streamlit application. The user enters a review, the app preprocesses it, loads the saved SimpleRNN model, and displays the sentiment, prediction confidence, and probability information.

```text
User
  ↓
Streamlit interface
  ↓
Review preprocessing
  ↓
Trained SimpleRNN model
  ↓
Prediction and confidence display
  ↓
Positive / Negative
```

The app also has example reviews, character and word counts, a clear-input control, a model-information sidebar, a pipeline explanation, and prediction history stored in the current Streamlit session. Reviews are not written to disk or an external database by the app.

## Project Features

- IMDB positive/negative sentiment classification.
- Saved SimpleRNN model with an Embedding layer.
- IMDB word-index-based text encoding and fixed-length sequence padding.
- Streamlit interface for entering and analyzing reviews.
- Prediction score and confidence display.
- Example review buttons and review length counts.
- Session-scoped prediction history with a clear-history control.
- Cached model and word-index loading.
- Training and preprocessing walkthroughs in Jupyter notebooks.

## Project Structure

```text
deep-learning-imdb-sentiment-rnn/
├── main.py                         # Root Streamlit sentiment-analysis app
├── simple_rnn_imdb.h5              # Trained model used by the root app
├── embedding.ipynb                 # Embedding concepts notebook
├── predection.ipynb                # Prediction walkthrough (filename as stored)
├── simpleRNN.ipynb                 # IMDB SimpleRNN training notebook
├── .vscode/
│   └── settings.json               # VS Code Python environment settings
└── simple_rnn_imdb/                # Nested copy of project materials
    ├── main.py                     # Nested Streamlit app
    ├── simple_rnn_imdb.h5          # Nested model copy
    ├── requirements.txt             # Dependency manifest in this folder
    ├── embedding.ipynb
    ├── prediction.ipynb
    └── simplernn.ipynb
```

The repository has no root-level `requirements.txt`; the manifest shown above is inside `simple_rnn_imdb/`. The root app expects the root-level `simple_rnn_imdb.h5` file next to `main.py`.

## Run the Streamlit App

Run commands from the repository root. The only dependency manifest in the repository is `simple_rnn_imdb/requirements.txt`; it pins TensorFlow to `2.15.0`, so use a Python version compatible with that TensorFlow release.

Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the listed dependencies and start the root app:

```powershell
python -m pip install -r simple_rnn_imdb/requirements.txt
streamlit run main.py
```

The app loads `simple_rnn_imdb.h5` from the repository root. On its first prediction, it also loads the IMDB word index through Keras.

## Implementation Notes and Limitations

- The root app's tokenizer uses lowercase plus whitespace splitting only; it does not remove punctuation. A token such as `amazing.` may therefore not match the word-index entry for `amazing`.
- The root app uses the full `imdb.get_word_index()` mapping and does not filter looked-up word ranks to the model's 10,000-entry embedding vocabulary. A known low-frequency word can consequently produce an ID outside the model's supported range. This README describes the current implementation and does not imply that every arbitrary review is guaranteed to process successfully.
- The current unknown-word fallback maps an absent token to `5` after the `+3` offset, as detailed above.
- The app's confidence percentage is based on the model output and should not be interpreted as certainty or as an independently measured accuracy.
- `simple_rnn_imdb/` contains a second copy of the app/model/notebooks and a requirements file. The root app command above specifically runs the root `main.py` and uses the root model file.
