import os

# Use Keras 2 so the exported model converts cleanly to TensorFlow.js.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import math
import numpy as np

dataset, info = tfds.load(
    "fashion_mnist",
    as_supervised=True,
    with_info=True,
    data_dir="data",
)

# print(info)

train_ds, test_ds = dataset["train"], dataset["test"]

class_names = info.features["label"].names

# print(class_names)

# Normalize the data (scale from 0-255 to 0-1)
def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255  # scales from 0-255 to 0-1
  return images, labels

# Normalize the training and test data with the function above
train_ds = train_ds.map(normalize)
test_ds = test_ds.map(normalize)

# Cache the data (keep it in memory instead of disk, faster training)
train_ds = train_ds.cache()
test_ds = test_ds.cache()

# Show the first image from the training data
for image, label in train_ds.take(1):
  break
image = image.numpy().reshape((28, 28))

# Draw
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(25)):
  image = image.numpy().reshape((28, 28))
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(image, cmap=plt.cm.binary)
  plt.xlabel(class_names[label])
plt.show()

# Create the model
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # 1 - grayscale
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax),  # for classification networks
])

# Compile the model
model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=["accuracy"],
)

# Number of examples in training and test sets (60k and 10k)
num_train_examples = info.splits["train"].num_examples
num_test_examples = info.splits["test"].num_examples

print(num_train_examples)
print(num_test_examples)

# Batching lets training with large amounts of data run more efficiently
BATCH_SIZE = 32

# Shuffle and repeat so the data is in random order and the network
# doesn't end up learning the order of things
train_ds = train_ds.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_ds = test_ds.batch(BATCH_SIZE)

# Train
history = model.fit(
  train_ds,
  epochs=5,
  steps_per_epoch=math.ceil(num_train_examples / BATCH_SIZE),
)

# Plot the loss function
plt.xlabel("Epoch #")
plt.ylabel("Loss magnitude")
plt.plot(history.history["loss"])
plt.show()

# Draw a grid of several predictions, marking correct (blue) or incorrect (red)
for test_images, test_labels in test_ds.take(1):
  test_images = test_images.numpy()
  test_labels = test_labels.numpy()
  predictions = model.predict(test_images)


def plot_image(i, predictions_array, true_labels, images):
  predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[..., 0], cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = "blue"
  else:
    color = "red"

  plt.xlabel(
    "{} {:2.0f}% ({})".format(
        class_names[predicted_label],
        100 * np.max(predictions_array),
        class_names[true_label],
    ),
    color=color,
  )


def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  bar_plot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  bar_plot[predicted_label].set_color("red")
  bar_plot[true_label].set_color("blue")


rows = 5
cols = 5
num_images = rows * cols
plt.figure(figsize=(2 * 2 * cols, 2 * rows))
for i in range(num_images):
  plt.subplot(rows, 2 * cols, 2 * i + 1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(rows, 2 * cols, 2 * i + 2)
  plot_value_array(i, predictions, test_labels)
plt.show()

# Test a single image
image = test_images[4]  # test_images only holds what was set in the block above
image = np.array([image])
prediction = model.predict(image)

print("Prediction: " + class_names[np.argmax(prediction[0])])

# Export the model to H5
model.save("exported_model.h5")





