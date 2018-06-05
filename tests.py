import keras
from keras_mlp import Keras_MLP
import numpy as np

x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

classifier = Keras_MLP(
                task="classification",
		layer_sizes=(100, 100, 100),
                activations = 'relu',
                dropout='Auto',
                alpha=0.00001*(2**1),
                batch_size=200,
                learning_rate_init=0.001,
                epochs=15,
                shuffle=True,
                loss_function = "categorical_crossentropy",
                metrics = ['binary_accuracy'],
                verbose=1,
                early_stopping=False,
                optimizer_name="adam",
                lr=0.001,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon=1e-08)

new_model = classifier.create_model(x_train, y_train)

for i in range(0, 100):
        new_model.fit(x_train, 
                      y_train, 
                      batch_size=classifier.batch_size, 
                      epochs=classifier.epochs, 
                      verbose=classifier.verbose, 
                      callbacks=classifier.used_callbacks, 
                      shuffle=classifier.shuffle)