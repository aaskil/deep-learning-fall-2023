"""
November 2023

@author: Askil Folgerø

This file contains the functions used in the training and testing of the neural network.
The functions are used in the main file, and are not intended to be run on their own.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pprint
import matplotlib.pyplot as plt
import datetime
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass, field
import os
import datetime
import csv
import torchinfo

@dataclass
class TrainingParams:
    # the network
    model: torch.nn.Module
    criterion: torch.nn.Module
    optimizer_string: str
    epochs_value: int
    batch_size: int

    # plot_data
    total_parameters: int = 0
    training_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    accuracies: List[float] = field(default_factory=list)
    test_accuracy: float = 0.0

    # optimizer params
    optimizer: Optional[torch.optim.Optimizer] = None
    use_scheduler: Optional[bool] = False
    scheduler_factor: Optional[float] = 0.1
    scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
    learning_rate_value: Optional[float] = None
    momentum_value: Optional[float] = None
    dampening: Optional[float] = None
    nesterov: Optional[bool] = None
    betas: Optional[Tuple[float, float]] = None
    momentum_decay: Optional[float] = None
    amsgrad: Optional[bool] = None
    eps: Optional[float] = None

    # regularization params
    dropout_value: Optional[float] = None
    regularization: Optional[str] = None
    weight_decay: Optional[float] = None

    # data
    train_data: Optional[torch.utils.data.Dataset] = None
    train_loader: Optional[torch.utils.data.DataLoader] = None
    validation_data: Optional[torch.utils.data.Dataset] = None
    validation_loader: Optional[torch.utils.data.DataLoader] = None
    test_data: Optional[torch.utils.data.Dataset] = None
    test_loader: Optional[torch.utils.data.DataLoader] = None

    # info
    path: str = None
    csv_path: str = path
    subfolder: str = None
    dense_numb: Optional[int] = None
    convu_numb: Optional[int] = None
    second_text_box: Optional[str] = None

    # precision recall
    true_positive: Optional[int] = None
    false_positive: Optional[int] = None
    true_negative: Optional[int] = None
    false_negative: Optional[int] = None

    def set_optimizer(self):
        if self.optimizer_string == "sgd":
            optimizer_params = {
                "params": self.model.parameters(),
            }
            if self.learning_rate_value is not None:
                optimizer_params["lr"] = self.learning_rate_value
            if self.momentum_value is not None:
                optimizer_params["momentum"] = self.momentum_value
            if self.weight_decay is not None:
                optimizer_params["weight_decay"] = self.weight_decay
            if self.nesterov is not None:
                optimizer_params["dampening"] = self.nesterov
            if self.dampening is not None:
                optimizer_params["dampening"] = self.dampening
            if self.weight_decay is not None:
                optimizer_params["nesterov"] = self.nesterov
            self.optimizer = optim.SGD(**optimizer_params)

        elif self.optimizer_string == "adagrad":
            optimizer_params = {
                "params": self.model.parameters(),
            }
            if self.learning_rate_value is not None:
                optimizer_params["lr"] = self.learning_rate_value
            if self.lr_decay is not None:
                optimizer_params["lr_decay"] = self.lr_decay
            if self.weight_decay is not None:
                optimizer_params["weight_decay"] = self.weight_decay
            if self.eps is not None:
                optimizer_params["eps"] = self.eps
            self.optimizer = optim.Adagrad(**optimizer_params)

        elif self.optimizer_string == "rmsprop":
            optimizer_params = {
                "params": self.model.parameters(),
            }
            if self.learning_rate_value is not None:
                optimizer_params["lr"] = self.learning_rate_value
            if self.momentum_value is not None:
                optimizer_params["momentum"] = self.momentum_value
            if self.alpha is not None:
                optimizer_params["alpha"] = self.alpha
            if self.eps is not None:
                optimizer_params["eps"] = self.eps
            if self.centered is not None:
                optimizer_params["centered"] = self.centered
            if self.weight_decay is not None:
                optimizer_params["weight_decay"] = self.weight_decay
            self.optimizer = optim.RMSprop(**optimizer_params)

        elif self.optimizer_string == "adam":
            optimizer_params = {
                "params": self.model.parameters(),
            }
            if self.learning_rate_value is not None:
                optimizer_params["lr"] = self.learning_rate_value
            if self.betas is not None:
                optimizer_params["betas"] = self.betas
            if self.eps is not None:
                optimizer_params["eps"] = self.eps
            if self.weight_decay is not None:
                optimizer_params["weight_decay"] = self.weight_decay
            if self.amsgrad is not None:
                optimizer_params["amsgrad"] = self.amsgrad
            self.optimizer = optim.Adam(**optimizer_params)

        elif self.optimizer_string == "nadam":
            optimizer_params = {
                "params": self.model.parameters(),
            }
            if self.learning_rate_value is not None:
                optimizer_params["lr"] = self.learning_rate_value
            if self.betas is not None:
                optimizer_params["betas"] = self.betas
            if self.eps is not None:
                optimizer_params["eps"] = self.eps
            if self.weight_decay is not None:
                optimizer_params["weight_decay"] = self.weight_decay
            if self.momentum_decay is not None:
                optimizer_params["momentum_decay"] = self.momentum_decay
            
            self.optimizer = optim.NAdam(**optimizer_params)

        else:
            raise ValueError("Unsupported optimizer type")

        if self.use_scheduler:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=self.scheduler_factor, patience=2, verbose=True
            )
        return self.optimizer

    def set_path(self):
        fc_input_size = self.model.dense_input_size()
        learning_rate = str(self.learning_rate_value).replace(".", "_")
        self.subfolder = f"{self.path}/fc_size_{fc_input_size}/batchsize_{self.batch_size}/lr_rate_{learning_rate}"

    def precision(self):
        denominator = self.true_positive + self.false_positive
        if denominator == 0:
            return -1  # Or some other value or raise an exception
        return self.true_positive / denominator

    def recall(self):
        denominator = self.true_positive + self.false_negative
        if denominator == 0:
            return -1  # Or some other value or raise an exception
        return self.true_positive / denominator

    def precision_negative(self):
        denominator = self.true_negative + self.false_negative
        if denominator == 0:
            return -1  # Or some other value or raise an exception
        return self.true_negative / denominator

    def recall_negative(self):
        denominator = self.true_negative + self.false_positive
        if denominator == 0:
            return -1  # Or some other value or raise an exception
        return self.true_negative / denominator


def count_layers(model):
    num_convs = sum(1 for layer in model.modules() if isinstance(layer, nn.Conv2d))
    num_denses = sum(1 for layer in model.modules() if isinstance(layer, nn.Linear))
    return num_convs, num_denses


def test_model(trainingparams: TrainingParams,):
    trainingparams.model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for inputs, targets in trainingparams.test_loader:

            if trainingparams.criterion._get_name() == "BCELoss":
                
                outputs = trainingparams.model(inputs).squeeze(1)
                predicted = torch.round(outputs)

            elif trainingparams.criterion._get_name() == "CrossEntropyLoss":
                outputs = trainingparams.model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)

            else:
                raise ValueError("Unsupported loss function")
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


def cnn_training_test_loop(trainingparams: TrainingParams):
    """this is a doc string
    intended use:
    create a instance for TrainingParams
    cnn_training_test_loop(trainingparams)
    """
    optimizer = trainingparams.set_optimizer()
    convu_numb, dense_numb = count_layers(trainingparams.model)
    trainingparams.convu_numb = convu_numb
    trainingparams.dense_numb = dense_numb

    train_loader = DataLoader(
        trainingparams.train_data, 
        batch_size=trainingparams.batch_size, 
        shuffle=True
    )
    validation_loader = DataLoader(
        trainingparams.validation_data,
        batch_size=trainingparams.batch_size,
        shuffle=False,
    )
    for epoch in range(trainingparams.epochs_value):
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        trainingparams.model.train()
        total_train_loss = 0
        total_validation_loss = 0
        correct = 0
        total = 0
        num_zeros = 0
        num_ones = 0


        for inputs, target in train_loader:
            optimizer.zero_grad()

            if trainingparams.criterion._get_name() == "BCELoss":
                output = trainingparams.model(inputs).squeeze(1)
                train_loss = trainingparams.criterion(output, target.float())

            elif trainingparams.criterion._get_name() == "CrossEntropyLoss":
                output = trainingparams.model(inputs)
                train_loss = trainingparams.criterion(output, target)
            else:
                raise ValueError("Unsupported loss function")

            train_loss.backward()
            total_train_loss += train_loss.item()
            optimizer.step()

        trainingparams.model.eval()
        with torch.no_grad():
            for inputs, targets in validation_loader:
                if trainingparams.criterion._get_name() == "BCELoss":
                    outputs = trainingparams.model(inputs).squeeze(1)
                    test_loss = trainingparams.criterion(outputs, targets.float())
                    predicted = torch.round(outputs)

                elif trainingparams.criterion._get_name() == "CrossEntropyLoss":
                    outputs = trainingparams.model(inputs)
                    _, predicted = torch.max(outputs.data, dim=1)
                    test_loss = trainingparams.criterion(outputs, targets)
                else:
                    raise ValueError("Unsupported loss function")


                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                num_zeros += (predicted == 0).sum().item()
                num_ones += (predicted == 1).sum().item()
                total_validation_loss += test_loss.item()

                true_positive += ((predicted == 1) & (targets == 1)).sum().item()
                false_positive += ((predicted == 1) & (targets == 0)).sum().item()
                true_negative += ((predicted == 0) & (targets == 0)).sum().item()
                false_negative += ((predicted == 0) & (targets == 1)).sum().item()

        trainingparams.true_positive = true_positive
        trainingparams.false_positive = false_positive
        trainingparams.true_negative = true_negative
        trainingparams.false_negative = false_negative

        average_train_loss = total_train_loss / len(train_loader)
        average_validation_loss = total_validation_loss / len(validation_loader)

        if trainingparams.use_scheduler:
            trainingparams.scheduler.step(average_train_loss)

        trainingparams.training_loss.append(average_train_loss)
        trainingparams.validation_loss.append(average_validation_loss)
        trainingparams.accuracies.append(correct / total)
        trainingparams.test_accuracy = test_model(trainingparams)

        print(
            f"Epoch: {(epoch+1):>2}  \t"
            f"Train Loss: {average_train_loss:.4f} \t"
            f"Test Loss: {average_validation_loss:.4f} \t"
            f"Test Accu: {correct / total:.2f} \t"
            f"cat: {num_zeros:3} \t"
            f"dog: {num_ones:3} \t"
            f"precision: {trainingparams.precision():.2f}, {trainingparams.precision_negative():.2f} \t"
            f"recall: {trainingparams.recall():.2f}, {trainingparams.recall_negative():.2f}"
        )
