# import os
# import openai
import tensorflow as tf
import keras
from keras.datasets import imdb
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from keras import models
from keras import layers
import jupyter


# from sentence_transformers import SentenceTransformer

def short_roberta(sentences):
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    embeddings = model.encode(sentences)
    print(embeddings)
    return embeddings


# roberta fully copied from https://huggingface.co/sentence-transformers/all-roberta-large-v1
def long_roberta(sentences):
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    # Sentences we want sentence embeddings for
    # sentences = ['This is an example sentence', 'Each sentence is converted']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
    model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


def as_book():
    def vectorize_sequences(sequences, dimension=10000):
        results = np.zeros((len(sequences), dimension))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
    x_train = vectorize_sequences(train_data)
    y_test = vectorize_sequences(test_data)
    return x_train, y_test


def get_data():
    def decode(data, reverse_word_index):
        decoded_train_data = []
        for review in data:
            decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in review])
            decoded_train_data.append(decoded_review)
        return decoded_train_data

    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=100)
    train_data = train_data[:int(round(len(train_data) / 2000))]
    train_labels = train_labels[:int(round(len(train_labels) / 2000))]
    test_data = test_data[:int(round(len(test_data) / 2000))]
    test_labels = test_labels[:int(round(len(test_labels) / 2000))]

    print(decode(train_data, reverse_word_index)[0])
    x_train = long_roberta(decode(train_data, reverse_word_index)).numpy()
    x_test = long_roberta(decode(test_data, reverse_word_index)).numpy()
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    return x_train, x_test, y_train, y_test
    # sentences = ['This is an example sentence', 'Each sentence is converted']
    # return sentences


def main():
    x_train, x_test, y_train, y_test = get_data()
    '''
    x_train = x_train[:len(x_train)/20]
    x_test = x_test[:len(x_test)/20]
    y_train = y_train[:len(y_train)/20]
    y_test = y_test[:len(y_test)/20]
    '''
    print(type(x_train))
    print(len(x_train))
    print(type(x_train[0]))
    print(len(x_train[0]))
    print(x_train[0])

    x_val = x_train[:500]
    partial_x_train = x_train[500:]
    y_val = y_train[:500]
    partial_y_train = y_train[500:]

    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape='1024'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    print('start compiling')

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=256, validation_data=(x_val, y_val))

    plot(history)


def plot(history):
    import matplotlib.pyplot as plt

    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
