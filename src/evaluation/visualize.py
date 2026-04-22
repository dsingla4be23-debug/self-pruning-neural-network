import matplotlib.pyplot as plt

def plot_gates(model):
    gates = model.get_all_gates().detach().cpu().numpy()

    plt.hist(gates, bins=50)
    plt.title("Gate Value Distribution")
    plt.xlabel("Gate Value")
    plt.ylabel("Frequency")
    plt.savefig("gate_distribution.png")
    plt.close()
