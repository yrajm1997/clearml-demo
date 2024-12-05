## NOTE: Make sure you have a queue named 'services' with atleast one worker serving it, before executing this file.

from clearml import PipelineDecorator
import tensorflow as tf
from clearml import Task, TaskTypes

parameters = {
    'epochs': 3,
    'neurons': 128,         # 128, 110
    'hidden_layers': 2,     # 2, 1
    'activation': 'relu',
    'optimizer': 'adam',
    'dropout_rate': 0.2     # 0.2, 0.3
}


# Load data
@PipelineDecorator.component(return_values=['x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test'], cache=True, task_type=TaskTypes.data_processing)
def load_data():
    from tensorflow.keras.datasets import mnist
    (train_images, train_labels), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize data
    train_images = train_images.reshape((len(train_images), 28*28)).astype("float32") / 255.0
    x_test = x_test.reshape((len(x_test), 28*28)).astype("float32") / 255.0

    x_train = train_images[10000:]
    y_train = train_labels[10000:]
    x_val = train_images[:10000]
    y_val = train_labels[:10000]
    return x_train, y_train, x_val, y_val, x_test, y_test

# Train Model
@PipelineDecorator.component(return_values=['model'], cache=True, task_type=TaskTypes.training)
def train_model(x_train, y_train, x_val, y_val, parameters):
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
    
    return model


# Evaluate model
@PipelineDecorator.component(return_values=['loss', 'accuracy'], task_type=TaskTypes.qc)
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy


@PipelineDecorator.pipeline(name="pipeline-1", project="MNIST Pipeline", version="0.1")
def executing_pipeline():
    print("Model hyperparameters:", parameters)
    x_train, y_train, x_val, y_val, x_test, y_test = load_data()
    model = train_model(x_train, y_train, x_val, y_val, parameters)
    loss, accuracy = evaluate_model(model, x_test, y_test)
    print(f"Accuracy: {accuracy} Loss: {loss}")


if __name__ == "__main__":

    # PipelineDecorator.run_locally()          # to run it locally (Pipeline job as well as steps, both will run locally)
    # OR
    PipelineDecorator.set_default_execution_queue('default')      # this will run the Pipeline job in 'services' queue and the steps in 'default' queue

    executing_pipeline()
    print("Process completed")
