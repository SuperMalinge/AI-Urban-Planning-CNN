import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread as mpl_imread
from skimage.transform import resize
from tensorflow.keras.losses import MeanSquaredError

np.random.seed(678)
tf.random.set_seed(5678)

class ConLayerLeft(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerLeft, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = self.add_weight(shape=(kernel_size, kernel_size, in_channels, out_channels),
                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))

    def call(self, inputs, stride=1):
        current_shape_size = inputs.shape
        output_shape = [tf.shape(inputs)[0],
                       int(current_shape_size[1]),
                       int(current_shape_size[2]),
                       self.out_channels]

        layer = tf.nn.conv2d(inputs, self.w, strides=[1, stride, stride, 1], padding='SAME')
        layerA = tf.nn.relu(layer)
        return layerA

class ConLayerRight(tf.keras.layers.Layer):
    def __init__(self, kernel_size, in_channels, out_channels):
        super(ConLayerRight, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = self.add_weight(shape=(kernel_size, kernel_size, out_channels, in_channels),
                                initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05))

    def call(self, inputs, stride=1, output=1):
        current_shape_size = inputs.shape
        output_shape = [tf.shape(inputs)[0],
                       int(current_shape_size[1]),
                       int(current_shape_size[2]),
                       self.out_channels]
        
        layer = tf.nn.conv2d_transpose(inputs, self.w, output_shape=output_shape,
                                     strides=[1, 1, 1, 1], padding='SAME')
        layerA = tf.nn.relu(layer)
        return layerA

# Data loading for urban satellite imagery
data_location = "./URBAN/training/satellite/"
train_data = []
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():
            train_data.append(os.path.join(dirName, filename))

data_location = "./URBAN/training/masks/"
train_data_gt = []
for dirName, subdirList, fileList in sorted(os.walk(data_location)):
    for filename in fileList:
        if ".tif" in filename.lower():
            train_data_gt.append(os.path.join(dirName, filename))

# Initialize arrays for RGB satellite images and masks
train_images = np.zeros(shape=(128, 256, 256, 3))
train_labels = np.zeros(shape=(128, 256, 256, 1))

# Load and preprocess images
for file_index in range(len(train_data)):
    train_images[file_index, :, :] = resize(mpl_imread(train_data[file_index]), (256, 256, 3))
    train_labels[file_index, :, :] = np.expand_dims(
        resize(mpl_imread(train_data_gt[file_index], as_gray=True), (256, 256)), axis=2
    )

# Normalize images
train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min() + 1e-8)
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min() + 1e-8)

# Training parameters
num_epochs = 100
init_lr = 0.0001
batch_size = 2

class RetinalCNN(tf.keras.Model):
    def __init__(self, layers):
        super(RetinalCNN, self).__init__()
        self.layer_list = layers
        
    def call(self, inputs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

# Initialize with 3 channels for RGB input
current_channels = 3

# Create layers list
layers = []

# Layer configurations for urban feature detection
layer_configs = [
    (32, 3), (32, 3), (32, 3),  # Enhanced initial feature extraction
    (64, 3), (64, 3), (32, 3),  # Building detection
    (64, 3), (64, 3), (32, 3),  # Road network detection
    (64, 3), (64, 3), (32, 3),  # Green space detection
    (64, 3), (64, 3), (128, 3), # Urban density patterns
    (96, 3), (48, 3), (24, 3),  
    (32, 3), (16, 3), (8, 3),
    (16, 3), (8, 3), (4, 3),
    (8, 3), (4, 3), (4, 3),
    (1, 3)  # Final output for urban segmentation
]

for i, (out_channels, kernel_size) in enumerate(layer_configs):
    if i == 25:
        layers.append(ConLayerRight(kernel_size, current_channels, out_channels))
    else:
        layers.append(ConLayerLeft(kernel_size, current_channels, out_channels))
    current_channels = out_channels

# Create and compile model
model = RetinalCNN(layers)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr),
             loss='mse',
             metrics=['mae'])

# Training loop with visualization
for iter in range(num_epochs):
    for current_batch_index in range(0, len(train_images), batch_size):
        current_batch = train_images[current_batch_index:current_batch_index + batch_size]
        current_label = train_labels[current_batch_index:current_batch_index + batch_size]
        
        with tf.GradientTape() as tape:
            predictions = model(current_batch)
            loss_value = tf.reduce_mean(tf.square(predictions - current_label))
        
        gradients = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        print(' Iter: ', iter, " Cost:  %.32f" % loss_value.numpy(), end='\r')
    
    print('\n-----------------------')
    train_images, train_labels = shuffle(train_images, train_labels)
    
    # Visualization every 2 epochs
    if iter % 2 == 0:
        test_example = train_images[:2]
        test_example_gt = train_labels[:2]
        sess_results = model(test_example)

        for idx in range(2):
            plt.figure()
            plt.imshow(test_example[idx])
            plt.axis('off')
            plt.title(f'Satellite Image {idx}')
            plt.savefig(f'urban_development/{iter}a_Satellite_Image_{idx}.png')

            plt.figure()
            plt.imshow(np.squeeze(test_example_gt[idx]), cmap='gray')
            plt.axis('off')
            plt.title(f'Urban Development Mask {idx}')
            plt.savefig(f'urban_development/{iter}b_Urban_Mask_{idx}.png')

            plt.figure()
            plt.imshow(np.squeeze(sess_results[idx]), cmap='gray')
            plt.axis('off')
            plt.title(f'Predicted Development {idx}')
            plt.savefig(f'urban_development/{iter}c_Predicted_Development_{idx}.png')

            plt.figure()
            plt.imshow(np.multiply(test_example[idx], np.repeat(test_example_gt[idx], 3, axis=2)))
            plt.axis('off')
            plt.title(f'Ground Truth Urban Overlay {idx}')
            plt.savefig(f'urban_development/{iter}d_Ground_Truth_Overlay_{idx}.png')

            plt.figure()
            plt.imshow(np.multiply(test_example[idx], np.repeat(sess_results[idx], 3, axis=2)))
            plt.axis('off')
            plt.title(f'Predicted Urban Overlay {idx}')
            plt.savefig(f'urban_development/{iter}e_Predicted_Overlay_{idx}.png')

            plt.close('all')

print("Training completed successfully!")
