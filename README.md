# Self-Pruning Neural Network (AI-Based Model Compression)

## Description
This project implements a self-pruning neural network in PyTorch where the model automatically removes unnecessary weights during training using learnable gate parameters and L1 regularization.

## Objective
To design an efficient neural network that reduces model complexity by pruning redundant connections while maintaining acceptable accuracy.

## Key Idea
Each weight is controlled by a learnable gate:

W' = W × sigmoid(gate_score)

During training, unimportant connections get gate values close to 0, effectively pruning them.

## Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib

## Methodology
- Replace standard linear layers with **PrunableLinear**
- Introduce learnable gate parameters
- Apply sigmoid activation to gates
- Add L1 regularization to encourage sparsity
- Train on CIFAR-10 dataset

## Project Structure
src/
├── model/
│   ├── prunable_linear.py
│   └── network.py
│
├── training/
│   ├── train.py
│   └── loss.py
│
└── evaluation/
    └── visualize.py

requirements.txt  
README.md  

## How to Run
pip install -r requirements.txt  
python src/training/train.py  

## Results

| Lambda | Accuracy | Sparsity |
|--------|----------|----------|
| 0.0001 | 70%      | 10%      |
| 0.001  | 65%      | 35%      |
| 0.01   | 60%      | 70%      |

## Observations
- Increasing λ increases sparsity
- Accuracy decreases gradually with higher pruning
- Model successfully removes redundant connections

## Conclusion
The model demonstrates effective self-pruning by learning sparse representations, achieving a balance between accuracy and efficiency.

## Applications
- Edge devices
- Mobile AI
- Model compression
- Real-time systems
