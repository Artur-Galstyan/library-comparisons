from torch import nn
import torch
from datahandler import get_data


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        logits = self.network(x)
        return logits


def main():
    train_dataloader, test_dataloader = get_data()
    model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    n_epochs = 125
    model.train()
    for epoch in range(n_epochs):
        for x, y in train_dataloader:
            pred = model(x.float())
            loss = loss_fn(pred, y.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 25 == 0:
            loss = loss.item()
            print(f"loss: {loss:>7f}")

    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in test_dataloader:
            pred = model(x.float())
            test_loss += loss_fn(pred, y.float()).item()
            correct += (
                (pred.argmax(axis=1, keepdims=True) == y).type(torch.float).sum().item()
            )
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


if __name__ == "__main__":
    main()
