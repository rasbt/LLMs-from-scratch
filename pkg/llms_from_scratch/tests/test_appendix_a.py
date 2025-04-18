# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

from llms_from_scratch.appendix_a import NeuralNetwork, ToyDataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_dataset():

    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])

    y_train = torch.tensor([0, 0, 0, 1, 1])
    train_ds = ToyDataset(X_train, y_train)

    len(train_ds) == 5
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )

    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    num_epochs = 3

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, labels) in enumerate(train_loader):

            logits = model(features)

            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
                  f" | Train/Val Loss: {loss:.2f}")

        model.eval()
        with torch.no_grad():
            outputs = model(X_train)

        expected = torch.tensor([
            [2.8569, -4.1618],
            [2.5382, -3.7548],
            [2.0944, -3.1820],
            [-1.4814, 1.4816],
            [-1.7176, 1.7342]
        ])
        torch.equal(outputs, expected)
