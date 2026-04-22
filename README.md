# self-pruning-neural-network
# Description

This project implements a self-pruning neural network in PyTorch where the model learns to remove unnecessary weights during training using learnable gate parameters.

# Objective

To design a neural network that dynamically prunes itself using L1 regularization on gate values, reducing model complexity while maintaining accuracy.

# Technologies Used

* Python
* PyTorch
* NumPy
* Matplotlib

# How It Works

* Each weight has a learnable **gate parameter**
* Gates are passed through a **sigmoid function**
* Final weights = weight × gate
* L1 regularization pushes gates toward 0 → pruning

# Project Structure

* model.py → prunable layer & network
* train.py → training loop
* utils.py → helper functions
* results/ → graphs & outputs


# Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 0.001  | 85%      | 20%      |
| 0.01   | 80%      | 50%      |
| 0.1    | 72%      | 78%      |

# Observations

* Higher lambda → more pruning
* Accuracy decreases slightly with high sparsity
* Model learns important connections only

