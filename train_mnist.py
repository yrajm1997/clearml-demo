import tensorflow as tf

from clearml import Task

task = Task.init(project_name='MNIST_project',
                 task_name='training_task_2')


parameters = {
    'epochs': 3,
    'neurons': 128,         # 128, 110
    'hidden_layers': 2,     # 2, 1
    'activation': 'relu',
    'optimizer': 'adam',
    'dropout_rate': 0.2     # 0.2, 0.3
}

task.connect(parameters)

# Load data
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (x_test, y_test) = mnist.load_data()

# Reshape and normalize data
train_images = train_images.reshape((len(train_images), 28*28)).astype("float32") / 255.0
x_test = x_test.reshape((len(x_test), 28*28)).astype("float32") / 255.0

x_train = train_images[10000:]
y_train = train_labels[10000:]
x_val = train_images[:10000]
y_val = train_labels[:10000]

# Model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(28*28,)))
for i in range(parameters['hidden_layers']):
    model.add(tf.keras.layers.Dense(parameters['neurons'], activation=parameters['activation']))
model.add(tf.keras.layers.Dropout(parameters['dropout_rate']))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=parameters['optimizer'],
              loss=loss_fn,
              metrics=['accuracy'])

from datetime import datetime
logdir = "logs/digits/" + datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks_list = [tf.keras.callbacks.TensorBoard(log_dir=logdir),
                  tf.keras.callbacks.ModelCheckpoint("mnist_model_checkpoint.keras", save_best_only=True)]

model.fit(x_train, y_train,
          epochs=parameters['epochs'],
          validation_data = (x_val, y_val),
          callbacks = callbacks_list
          )

loss, acc = model.evaluate(x_test, y_test, verbose=2)

print(f"Accuracy: {acc} Loss: {loss}")

# task.get_logger.report_scaler(...)
