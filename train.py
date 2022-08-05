import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import transformer_model.hparams as hparams
import matplotlib.pyplot as plt
from IPython.display import clear_output
from transformer_model.model import transformer_model
from data.process_dataset import process_data


strategy = hparams.tpu_strategy()


class CustomLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super(CustomLearningRate, self).__init__()
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        var1 = tf.math.rsqrt(step)
        var2 = step * self.warmup_steps ** -1.5

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(var1, var2)


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


def accuracy(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, hparams.max_length - 1))

    return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)


class ModelPlot(tf.keras.callbacks.Callback):
    def __init__(self):
        self.metrics = {}

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        for metric in logs:
            self.metrics[metric] = []

    def on_epoch_end(self, epoch, logs=None):
        # Storing metrics
        if logs is None:
            logs = {}
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        metrics = [x for x in logs if 'val' not in x]

        f, axs = plt.subplots(1, len(metrics), figsize=(15, 5))
        clear_output(wait=True)

        for i, metric in enumerate(metrics):
            axs[i].plot(range(1, epoch + 2),
                        self.metrics[metric],
                        label=metric)
            if logs['val_' + metric]:
                axs[i].plot(range(1, epoch + 2),
                            self.metrics['val_' + metric],
                            label='val_' + metric)

            axs[i].legend()
            axs[i].grid()

        plt.tight_layout()
        plt.savefig("data/training_plots/plot.png")


def main():
    tf.keras.utils.set_random_seed(1234)
    tf.keras.backend.clear_session()

    dataset_train, dataset_val, vocab_size, tokenizer = process_data()

    callbacks_plot = [ModelPlot()]

    optimizer = tf.keras.optimizers.Adam(
        CustomLearningRate(d_model=hparams.d_model),
        beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    with strategy.scope():
        model = transformer_model(
            vocab_size=vocab_size,
            num_layers=hparams.num_layers,
            units=hparams.units,
            d_model=hparams.d_model,
            num_heads=hparams.num_heads,
            dropout=hparams.dropout
        )

        model.compile(optimizer, loss=loss_function, metrics=[accuracy])

    model.summary()

    model.fit(dataset_train, epochs=hparams.epochs, validation_data=dataset_val, callbacks=callbacks_plot)

    print("\nTraining finished! saving as {}.".format(hparams.output_dir))
    model.save(hparams.output_dir, include_optimizer=False)

    tf.keras.backend.clear_session()


if __name__ == '__main__':
    main()
