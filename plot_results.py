"""
Yuli Tshuva
Plot the results
"""

from os.path import join
import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Set fonts
rcParams["font.family"] = "Times New Roman"

OUTPUT_PATH = "output"
DATA_SET = "cora"
SAVE_DIR = join(OUTPUT_PATH, DATA_SET)
DICT_PATH = join(SAVE_DIR, "status_report.pkl")
TEST_ACC = 0.842
TEST_LOSS = 0.6526

colors = ["dodgerblue", "hotpink"]

# Read the status report
with open(DICT_PATH, "rb") as file:
    status_report = pickle.load(file)

# plot loss
plt.plot(status_report[1::4], label='Training Loss', color=colors[0])
plt.plot(status_report[::4], label='Validation Loss', color=colors[1], linewidth=2.2)
plt.plot([0, len(status_report)//4], [TEST_LOSS, TEST_LOSS],
         color="turquoise", linestyle="dashed", label="Test Loss", linewidth=2.5)
plt.legend()
plt.title('Loss Plot', fontsize=18)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.savefig(join(SAVE_DIR, 'loss_plot.png'))
plt.show()

# plot accuracy
plt.plot(status_report[3::4], label='Training Accuracy', color=colors[0])
plt.plot(status_report[2::4], label='Validation Accuracy', color=colors[1], linewidth=2)
plt.plot([0, len(status_report)//4], [TEST_ACC, TEST_ACC], color="turquoise",
         linestyle="dashed", label="Test Accuracy", linewidth=2.5)
plt.legend()
plt.title('Accuracy Plot', fontsize=18)
plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Accuracy", fontsize=14)
plt.yticks(np.arange(0, 1.01, 0.1))
plt.savefig(join(SAVE_DIR, 'accuracy_plot.png'))
plt.show()
