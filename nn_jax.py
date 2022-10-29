import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from datahandler import get_data


def relu(x):
    return jnp.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))


def init_random_layer(key, n_in, n_out):
    w_key, b_key = random.split(key)
    weight = random.normal(w_key, shape=(n_in, n_out)) * 0.1
    bias = random.normal(b_key, shape=(n_out,)) * 0.1
    return weight, bias


def init_params(key, layers: list):
    params = []
    keys = random.split(key, len(layers))
    for layer, key in zip(layers, keys):
        params.append(init_random_layer(key, layer[0], layer[1]))
    return params


def get_activations():
    return [relu, relu, sigmoid]


def predict(params, x):
    a = x
    for idx, (w, b) in enumerate(params):
        z = a @ w + b
        a = get_activations()[idx](z)

    return a


def loss(params, x, y):
    preds = predict(params, x)
    return jnp.mean((preds - y) ** 2)


@jit
def update(params, x, y, learning_rate=0.1):
    grads = grad(loss)(params, x, y)
    return [
        (w - learning_rate * dw, b - learning_rate * db)
        for (w, b), (dw, db) in zip(params, grads)
    ]


def get_current_accuracy(params, test_dataloader):
    correct = 0
    total_counter = 0
    for x, y in test_dataloader:
        x = x.numpy()
        y = y.numpy()
        a = predict(params, x)
        pred = jnp.argmax(a, axis=1, keepdims=True)
        y = jnp.argmax(y, axis=1, keepdims=True)
        correct += (pred == y).sum()
        total_counter += len(x)
    accuracy = correct / total_counter
    return accuracy


def main():
    train_dataloader, test_dataloader = get_data()
    layers = [(13, 32), (32, 32), (32, 2)]
    key = random.PRNGKey(42)
    params = init_params(key, layers)
    n_epochs = 125
    for epoch in range(n_epochs):
        current_loss = 0
        for x, y in train_dataloader:
            x = x.numpy()
            y = y.numpy()
            current_loss += loss(params, x, y)
            params = update(params, x, y)
        print(f"Loss = {current_loss / len(train_dataloader)}")
        print(f"Accuracy = {get_current_accuracy(params, test_dataloader)}")


if __name__ == "__main__":
    main()
