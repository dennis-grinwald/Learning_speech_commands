import numpy as np
import matplotlib.pyplot as plt

train_curve_path = "results/128_hidden/1fc/train_loss.npy"
test_curve_path = "results/128_hidden/1fc/test_accuracy.npy"

training_loss = np.load(train_curve_path)
test_accuracy = np.load(test_curve_path)

fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax1.plot(training_loss, label="Training loss")
ax1.set_xlabel("No. epochs")
ax1.set_ylabel("Training Loss")

ax2 = fig.add_subplot(1,2,2)
ax2.plot(test_accuracy, label="Testing Accuracy")
ax2.set_xlabel("No. epochs")
ax2.set_ylabel("Test Accuracy[%]")

plt.show()
