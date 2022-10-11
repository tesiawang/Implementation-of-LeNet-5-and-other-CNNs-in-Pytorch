import numpy as np
import matplotlib.pyplot as plt


train_loss = np.load('EP20_LR0.001_BS512_train_loss.npy')
n_ep = np.arange(20) + 1
plt.plot(n_ep, train_loss, 'o-', linewidth=2.0, )
plt.xticks(n_ep)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.savefig('train_loss.png')

