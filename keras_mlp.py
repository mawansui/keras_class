""" Это – класс для обучения нейросети на библиотеке Keras. """

from keras import optimizers, regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
import pickle
from get_optimizer import get_optimizer
import os.path
# импортируем список всех возможных опций
from choose_parameters import options
from define_dropout import define_dropout

class Keras_MLP():
    def __init__(self,
                 task, # ввод названия задачи - - - - - - - - - - - - - - - - - 
                 layer_sizes, # только скрытые слои - - - - - - - - - - - - - - 
                 activations, 
                 dropout, # указываем, нужен ли дропаут и если да, то сколько -
                 # дропаут должен идти после каждого скрытого слоя 
                 alpha, 
                 batch_size, 
                 learning_rate_init, 
                 epochs,
                 shuffle,
                 loss_function, 
                 metrics, 
                 verbose, 
                 early_stopping,
                 optimizer_name,
                 **kwargs):
        self.task = task # ввод названия задачи - - - - - - - - - - - - - - - - 
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.dropout = dropout
        # dropout может быть строкой ("Auto"), массивом с 1 числом или несколькими
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.epochs = epochs
        self.shuffle = shuffle
        self.loss_function = loss_function
        self.metrics = metrics # = = = = = = = = = = = = = = = = = = = = = = = =
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.optimizer_name = optimizer_name
        for key, value in kwargs.items():
          setattr(self, key, value)

    def fit(self, x_train, y_train):
        """
            Принимает данные. 
            Создаёт модель на основании параметров, переданных в __init__. 
            Обучает модель.
            Возвращает обученную модель.
        """

        try:
            # массив. 1 – функция активации для последнего слоя, 2 - функция ошибки
            chosen_task = options[self.task]
        except Exception:
            print("No such task found in choose_parameters.py. Reverting to default")
            chosen_task = options["default"]

        # Создаём пустую модель (шампур, на который потом будем нанизывать слои)
        model = Sequential()

        # Сохраняем shape переданных данных
        # x_train_shape нужен для самого первого слоя (размерность вход. данных)
        # y_train_shape нужен для последнего слоя (размерность выход. данных)
        x_train_shape = int(x_train.shape[1])
        y_train_shape = int(y_train.shape[1])

        # Проверяем, верно ли задан размер последнего слоя, переданный при
        # инициализации класса, потому что он не может быть меньше y_train_shape
        # if self.layer_sizes[-1] != y_train_shape:
        #     print('Error building the network ( in keras_mlp.fit() ). '
        #           'Number of neurons in the last layer should be equal to '
        #           'y_train_shape.\n'
        #           'Ошибка при создании сети ( в keras_mlp.fit() ).'
        #           'Число нейронов в последнем слое нейросети должно равняться '
        #           'y_train_shape')
        # else:

        # возращает массив с параметрами для дропаута (может быть None, though!)
        used_dropout = define_dropout(self.dropout, self.layer_sizes)
        indices = [i for i in range(0, len(self.layer_sizes))]

        for index, layer_size, dropout in zip(indices, self.layer_sizes, used_dropout):
            if index == 0:
                model.add(Dense(layer_size, 
                                input_dim=x_train_shape,
                                kernel_regularizer=regularizers.l2(self.alpha)))
                model.add(Activation(self.activations[index]))
                model.add(Dropout(dropout))
            else:
                model.add(Dense(layer_size, 
                                kernel_regularizer=regularizers.l2(self.alpha)))
                model.add(Activation(self.activations[index]))
                model.add(Dropout(dropout))


        # отдельно после всего добавляем последний слой. Размер последнего слоя
        # равен размерности y_train. К нему же добавляем активатор по выбранной 
        # задаче
        model.add(Dense(y_train_shape, 
                  kernel_regularizer=regularizers.l2(self.alpha)))
        model.add(Activation(chosen_task[0]))



        # если размер верный, то создать слой с заданным числом нейронов
        # и после него добавить заданную активацию;
        # а если этот слой – первый, заодно уточнить размер входящих данных
            # for index, layer_size in enumerate(self.layer_sizes):
            #     if index == 0:
            #         # если первый слой, то добавить нейроны и активатор
            #         model.add(Dense(layer_size, 
            #                         input_dim=x_train_shape,
            #                         kernel_regularizer=regularizers.l2(self.alpha)))
            #         model.add(Activation(self.activations[index]))
            #     elif index == len(layer_sizes)-1:
            #         # если последний слой, добавить нейроны и активатор, 
            #         # соответствующий задаче - - - - - - - - - - - - - - - - - -
            #         model.add(Dense(layer_size,
            #                         kernel_regularizer=regularizers.l2(self.alpha)))
            #         # добавляем активатор для выбранной задачи
            #         model.add(Activation(chosen_task[0]))
            #     else:
            #         # если не первый и не последний, добавить нейроны и активатор
            #         model.add(Dense(layer_size,
            #                         kernel_regularizer=regularizers.l2(self.alpha)))
            #         model.add(Activation(self.activations[index]))

            # Оптимизатор – функция, изменяющая веса связей между нейронами;
            # в sk-learn называется solver

        chosen_optimizer = None 

        # Возможны два пути:
        #
        # 1 путь. Собираем в один словарь все переменные класса
        #         и передаём их в get_optimizer(), где kwargs.get()
        #         всё равно среди этой кучи найдет нужные ей значения.
        #
        # 2 путь. Из всех переменных класса выделяем только последние
        #         параметры (**kwargs), которые затем передаём в 
        #         get_optimizer(), где kwargs.get() тоже будет искать
        #         нужные ей параметры, но в гораздо меньшей куче
        #
        # Была проведена проверка скорости выполнения обоих вариантов
        # с помощью стандартной функции timeit() на 10000 повторах.
        # См. файл kwargs_get_speedtest.py для подробностей.
        # 
        # Результаты:
        # - Словарь со всеми переменными класса:  155.58 сек
        # - Словарь только с нужными переменными: 155.70 сек
        # 
        # Вывод: оба варианта эквивалентны, поэтому оставляю первый
        # вариант как самый удобочитаемый и экономящий место (1 строка
        # вместо 5)

        chosen_optimizer = get_optimizer(self.optimizer_name, self.__dict__)
        

        # loss function – это то, что изменяется во время обучения (???)
        # metrics - показывает качество обучения сети, сравнивания 
        # предсказанные и верные значения


        # loss – функция ошибки; здесь нужен F2-score?
        # наверное, придется сделать custom metric (https://keras.io/metrics/)
        # metrics! нужны разные; добавить стандартные
        model.compile(loss=chosen_task[1], # функция ошибки под задачу - - -
                      optimizer=chosen_optimizer,
                      metrics=chosen_task[2]) # метрика под задачу - - - - -
        
        print("Model Summary:\n\n")
        model.summary()

        # Так называемые колл-бэки Keras принимает только в виде массива
        used_callbacks = []

        # На тот случай, если вдруг понадобится поддержка EarlyStopping,
        # код честно нашел где-то на гитхабе в обсуждениях.
        if self.early_stopping == True:
            early_stopping_callback = EarlyStopping(monitor="value_loss")
            used_callbacks.append(early_stopping_callback)

        # Нужно для большей совместимости с прежним кодом
        if self.batch_size == "auto":
            self.batch_size = 200

        # Возможно пригодится; тоже нашел на гитхабе где-то в обсуждениях. 
        # new_y_train = keras.utils.np_utils.to_categorical()

        model.fit(x_train, 
                  y_train,
                  batch_size = self.batch_size,
                  epochs = self.epochs,
                  verbose = self.verbose,
                  callbacks = used_callbacks,
                  shuffle = self.shuffle)

        return model
    
    def partial_fit(self, x_train, y_train):
    	# метод для обучения сети в несколько заходов
    	# если файл с обученной моделью уже существует,
    	keras_model_filename="trained_keras_model.h5" # не надо хардкодить!
    	if os.path.isfile(keras_model_filename):
    		# загрузить его
    		model = load_model(keras_model_filename)
    		# дообучить модель
    		model.fit(x_train, y_train)
    		# сохранить модель
    		model.save(keras_model_filename)
    	else:
    		model = self.fit(x_train, y_train)
    		model.save(keras_model_filename)