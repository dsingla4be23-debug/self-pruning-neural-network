import torch.nn.functional as F

def compute_loss(model, outputs, targets, lambda_):
    classification_loss = F.cross_entropy(outputs, targets)
    gates = model.get_all_gates()
    sparsity_loss = gates.sum()
    return classification_loss + lambda_ * sparsity_loss
