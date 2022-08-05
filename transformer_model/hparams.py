import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def tpu_strategy():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU {}".format(tpu.cluster_spec().as_dict()["worker"]))
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    return strategy


strategy = tpu_strategy()

max_length = 40
batch_size = 64 * strategy.num_replicas_in_sync
num_layers = 2
d_model = 256
num_heads = 8
units = 512
dropout = 0.1
epochs = 125
output_dir = "transformer_model/chatbot_saved_model"
