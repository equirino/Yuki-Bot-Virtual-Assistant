import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import transformer_model.hparams as hparams
import tensorflow as tf
import tensorflow_datasets as tfds
from data.create_dataset import filter_sentence


def comment_response(from_file, to_file):
    inputs, outputs = [], []
    with open(from_file, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            inputs.append(line)
    with open(to_file, 'r', encoding='utf-8') as output_file:
        for line in output_file:
            outputs.append(line)

    return inputs, outputs


def tokenizer_filterer(start_token, end_token, inputs, outputs, tokenizer):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = start_token + tokenizer.encode(sentence1) + end_token
        sentence2 = start_token + tokenizer.encode(sentence2) + end_token

        if len(sentence1) <= hparams.max_length and len(sentence2) <= hparams.max_length:
            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs,
                                                                     maxlen=hparams.max_length, padding="post")
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs,
                                                                      maxlen=hparams.max_length, padding="post")

    return tokenized_inputs, tokenized_outputs


def process_data():
    train_set_convo1 = "data/datasets/train_set.from"
    train_set_convo2 = "data/datasets/train_set.to"
    val_set_convo1 = "data/datasets/val_set.from"
    val_set_convo2 = "data/datasets/val_set.to"

    comments_train, responses_train = comment_response(train_set_convo1, train_set_convo2)
    comments_val, responses_val = comment_response(val_set_convo1, val_set_convo2)

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(comments_train + responses_train,
                                                                          target_vocab_size=2**13)

    start_token = [tokenizer.vocab_size]
    end_token = [tokenizer.vocab_size + 1]
    vocab_size = tokenizer.vocab_size + 2

    comments_train, responses_train = tokenizer_filterer(start_token, end_token, comments_train,
                                                         responses_train, tokenizer)
    comments_val, responses_val = tokenizer_filterer(start_token, end_token,
                                                     comments_val, responses_val, tokenizer)

    # print('Vocab size: {}'.format(vocab_size))
    # print('Number of samples: {}'.format(len(comments_train)))

    dataset_train = tf.data.Dataset.from_tensor_slices(
        ({"inputs": comments_train, "dec_inputs": responses_train[:, :-1]}, responses_train[:, 1:])
    )

    dataset_val = tf.data.Dataset.from_tensor_slices(
        ({"inputs": comments_val, "dec_inputs": responses_val[:, :-1]}, responses_val[:, 1:])
    )

    dataset_train = dataset_train.cache()
    dataset_train = dataset_train.shuffle(len(comments_train))
    dataset_train = dataset_train.batch(hparams.batch_size)
    dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)

    dataset_val = dataset_val.cache()
    dataset_val = dataset_val.shuffle(len(comments_val))
    dataset_val = dataset_val.batch(hparams.batch_size)
    dataset_val = dataset_val.prefetch(tf.data.experimental.AUTOTUNE)

    # print(dataset_train)

    return dataset_train, dataset_val, vocab_size, tokenizer
