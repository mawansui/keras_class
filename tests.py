import keras
from keras_mlp import Keras_MLP
import numpy as np
import pickle

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
                verbose=1,
                early_stopping=False,
                optimizer_name="adam", # !
                lr=0.001,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon=1e-08)

new_model = classifier.create_model(x_train.shape[1], y_train.shape[1])
print(new_model.to_json())

# Согласно этому треду (https://github.com/keras-team/keras/issues/4446), 
# << successive calls to fit will incrementally train the model >>

# for i in range(0, 50):
#         new_model.fit(x_train, 
#                       y_train, 
#                       batch_size=classifier.batch_size, 
#                       epochs=classifier.epochs, 
#                       verbose=classifier.verbose, 
#                       callbacks=classifier.used_callbacks, 
#                       shuffle=classifier.shuffle)

# print("\nPickling started\n")
# filename = "pickled_nn"
# outfile = open(filename, 'wb')
# pickle.dump(new_model, outfile)
# outfile.close()

# print("\nPickling finished\n")
# print("Now to the unpickling part.\n")

# infile = open(filename, 'rb')
# restored_model = pickle.load(infile)
# infile.close()

# print("\nUnpickling is done!\n")
# print("Trying to continue learning:\n")

for i in range(0, 50):
  hist = new_model.fit(x_train, 
                y_train, 
                batch_size=classifier.batch_size, 
                epochs=classifier.epochs, 
                verbose=classifier.verbose, 
                callbacks=classifier.used_callbacks, 
                shuffle=classifier.shuffle)
  print(hist.history)