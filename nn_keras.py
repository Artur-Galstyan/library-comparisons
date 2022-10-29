import tensorflow as tf
from datahandler import get_data
import torch
import numpy as np


def main():
    train_dataloader, test_dataloader = get_data()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(13,)),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=["accuracy"],
    )

    n_epochs = 125
    for epoch in range(n_epochs):
        for x, y in train_dataloader:
            model.fit(x.numpy(), y.numpy(), verbose=0)

    test_acc = 0
    total = 0
    for x, y in test_dataloader:
        test_loss, test_acc = model.evaluate(x.numpy(), y.numpy(), verbose=0)
        total += test_acc
    print(f"Test accuracy: {total/len(test_dataloader)}")


if __name__ == "__main__":
    main()
