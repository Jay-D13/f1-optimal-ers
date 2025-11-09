import numpy as np
import matplotlib.pyplot as plt

def plot_traces(s, result, label_prefix="opt"):
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(s, result["v"], label=f"{label_prefix} v")
    axs[0].set_ylabel("v [m/s]")

    axs[1].plot(s, result["soc"], label=f"{label_prefix} soc")
    axs[1].set_ylabel("SOC")

    axs[2].plot(s[:-1], result["P_ice"], label=f"{label_prefix} P_ice")
    axs[2].plot(s[:-1], result["P_k"], label=f"{label_prefix} P_k")
    axs[2].set_ylabel("Power [W]")

    axs[3].plot(s, np.cumsum(np.concatenate([[0], np.diff(s)/np.maximum(result["v"], 1e-3)])), label="time")
    axs[3].set_ylabel("t [s]")
    axs[3].set_xlabel("s [m]")

    for ax in axs: ax.grid(True); ax.legend()
    plt.tight_layout(); plt.show()