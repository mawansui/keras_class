# Шаг 0. Перед началом работы нужно установить на компьютер TensorFlow и Keras
# (https://keras.io/#installation). Если есть желание использовать partial fit
# и просто контроллировать процессы сохранения-открытия файлов модели, нужно 
# установить h5py (http://docs.h5py.org/en/latest/build.html)

# Шаг 1. Импортировать класс:

from keras_mlp import Keras_MLP

# Шаг 2. Создать экземпляр класса.
#
# Task может быть любым, главное прописать его в файле choose_parameters.py.
# Там для каждой задачи (regression, classification, ...) прописывается функция
# активации на последнем (выходном) слое нейросети, функция ошибки для всей 
# нейросети и массив метрик, по которым будет смотреться качество обучения
#
# layer_sizes – кортеж с количеством нейронов на каждом скрытом слое
#
# activations – в ближайшее время сделаю всё так же удобно как с Dropout,
# 				но пока нужно указать массив функций активаций такой же длины,
#				как кортеж слоёв
#
# dropout – "Auto": После каждого скрытого слоя будет стоять дропаут-слой с 
#					дефолтным параметром 0.5 (меняется в файле define_dropout.py)
#			Список с 1 числом: После каждого скрытого слоя будет стоять дропаут-
#							   слой с таким параметром
#			Список с числами по длине такой же, как кортеж: После каждого скрытого
#							   слоя будет стоять дропаут-слой с переданным из списка
#							   соответствующим значением
#
# Остальное всё довольно стандартно. 
# loss_function и metrics задаются в файле choose_parameters в зависимости от 
# задачи, так что в следующей версии они будут убраны из явного объявления класса. 
#
# После параметра optimizer_name идут все параметры, которые требуются для 
# выбранного оптимизатора, согласно документации Keras (https://keras.io/optimizers/)

classifier = Keras_MLP(
                task="classification",
		layer_sizes=(100, 100, 100),
                activations = ['relu'],
                dropout=[0],
                alpha=0.00001*(2**1),
                batch_size=200,
                learning_rate_init=0.001,
                epochs=15,
                shuffle=True,
                verbose=1,
                early_stopping=False,
                optimizer_name="adam",
                lr=0.001,
                beta_1 = 0.9,
                beta_2 = 0.999,
                epsilon=1e-08,
                loss_function = "categorical_crossentropy",
                metrics = ['binary_accuracy'])

# Шаг 3. Выполнить метод fit() у этого класса. 
#		 Метод принимает 2 параметра: 
# 		 	1. Numpy-массив тренировочных данных
#			2. Numpy-массив лейблов к этим данным
# 	  	 Метод возвращает модель, поэтому нужно эту модель 
#		 присвоить новой переменной:

trained_model = classifier.fit(x_train_data, y_train_data)

# Шаг 4. Выполнить метод predict(). Он принимает Numpy-массив тестовых данных.
# 		 Возвращает Numpy-массив предсказанных лейблов.

predicted_y = trained_model.predict(x_test_data)

# Что осталось доделать:
# - - - - - - - - - - - 
# 1. (+) Автоматический подбор функций активаций для скрытых слоёв как для дропаута 
#        сделано
# 2. (+) Убрать явное присвоение loss_function и metrics при объявлении класса
# 3. Добавить возможность выбирать, сохранять ли модель и если да, то под каким
#    именем
# 4. Сохранение недообученной модели (partial_fit) в оперативной памяти, без
#    записи на диск.
# 5. Проверка моделью типа входных данных, преобразование их к numpy-массивам
# 6. (+) Выводить исключения, а не просто что-то печатать в консоль
# 7. Добавить возможность по-человечески включать-выключать dropout