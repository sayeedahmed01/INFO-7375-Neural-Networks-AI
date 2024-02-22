import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

sample_size = 100
indices = np.random.choice(range(x.shape[0]), sample_size, replace=False)
x_sample = x[indices]
y_sample = y[indices]

x_train, x_temp, y_train, y_temp = train_test_split(x_sample, y_sample, test_size=0.30, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.50, random_state=42)

def save_images(images, labels, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i, image in enumerate(images):
        plt.imsave(f"{dirname}/{i}_label_{labels[i]}.png", image, cmap='gray')

train_dir = 'mnist_data/train'
val_dir = 'mnist_data/validation'
test_dir = 'mnist_data/test'

save_images(x_train, y_train, train_dir)
save_images(x_val, y_val, val_dir)
save_images(x_test, y_test, test_dir)

print("Datasets have been saved to the mnist_data folder.")
