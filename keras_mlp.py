""" Класс для обучения нейросети на библиотеке Keras. """

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from helpers.get_optimizer import get_optimizer
# импортируем список всех возможных опций
from helpers.choose_parameters import options
# функция, которая подбирает количество и параметры слоёв дропаута в зависимости
# от того, что передано при инициализации класса
from helpers.define_dropout import define_dropout
# внешняя функция, которая подбирает количество функций активации в зависимости
# от того, что передано при инициализации класса
from helpers.define_activations import define_activations

class Keras_MLP():
    def __init__(self,
                 task,
                 layer_sizes,
                 activations, 
                 dropout,
                 alpha, 
                 batch_size, 
                 learning_rate_init, 
                 epochs,
                 shuffle,
                 verbose, 
                 early_stopping,
                 optimizer_name,
                 **kwargs):
        self.task = task
        self.layer_sizes = layer_sizes
        self.activations = activations
        # activations может быть "Auto", "relu", ["relu", "relu"]
        self.dropout = dropout
        # dropout может быть строкой ("Auto"), массивом с 1 числом или несколькими
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.epochs = epochs
        self.shuffle = shuffle
        self.verbose = verbose
        self.early_stopping = early_stopping
        self.optimizer_name = optimizer_name
        for key, value in kwargs.items():
          setattr(self, key, value)

    def create_model(self, x_train, y_train):
        """
            Метод принимает все параметры, переданные классу при инициализации,
            и создаёт новую пустую модель, на них основываясь.
            Затем он возвращает эту модель.
        """

        # Проверка – есть ли задача, переданная в параметр task, в файле
        # choose_parameters.py? Если есть, взять требуемые для задачи значения 
        # из этого файла, а если нет – взять из этого файла default-значения.
        
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

        # Возращает массив с параметрами дропаута для каждого скрытого слоя
        used_dropout = define_dropout(self.dropout, self.layer_sizes)

        # Возращает массив с названиями функции активации для каждого скрытого слоя
        used_activations = define_activations(self.activations, self.layer_sizes)

        # Создает массив с индексами всех слоёв, чтобы можно было использовать 
        # в цикле for
        indices = [i for i in range(0, len(self.layer_sizes))]

        # Главный цикл, который на "шампур" нанизывает слои нужного размера 
        # с соответствующими им значениями функций активаций и параметров 
        # дропаут-слоёв
        for index, layer_size, dropout, activation in zip(indices, 
                                                          self.layer_sizes, 
                                                          used_dropout, 
                                                          used_activations):

            # В случае первого слоя нужно конкретно указать его размерность
            if index == 0:
                model.add(Dense(layer_size, 
                                input_dim=x_train_shape,
                                kernel_regularizer=regularizers.l2(self.alpha)))
                model.add(Activation(activation))
                model.add(Dropout(dropout))
            else:
                model.add(Dense(layer_size, 
                                kernel_regularizer=regularizers.l2(self.alpha)))
                model.add(Activation(activation))
                model.add(Dropout(dropout))


        # Отдельно после всего добавляем последний слой. Размер последнего слоя
        # равен размерности y_train. К нему же добавляем активатор по выбранной 
        # задаче
        model.add(Dense(y_train_shape, 
                  kernel_regularizer=regularizers.l2(self.alpha)))
        model.add(Activation(chosen_task[0]))

        # Внешняя функция get_optimizer() вернет в перменную chosen_optimizer
        # выбранный оптимизатор на основании того, что пользователь указал при
        # инициализации класса
        chosen_optimizer = None
        chosen_optimizer = get_optimizer(self.optimizer_name, self.__dict__)

        # Проверяем, указаны ли в **kwargs параметры loss_function и metrics,
        # чтобы не привязываться к файлу choose_parameters.py.
        # Их передаём в виде массива, потому что так было сделано в прошлой 
        # версии – это в принципе можно изменить. TODO.
        parameters_to_compile = []

        # Если при инициализации класса была указана loss_function,
        # добавить её в этот пустой массив parameters_to_compile[:],
        # а если нет, то использовать ту функцию, которая указана в файле
        # choose_parameters.py для выбранной при инициализации задачи (task)
        if self.loss_function:
            parameters_to_compile.append(self.loss_function)
        else:
            parameters_to_compile.append(chosen_task[1])
        
        # Если при инициализации класса были указаны metrics (в виде массива!),
        # добавить их в этот пустой массив parameters_to_compile[:],
        # а если нет, то использовать тот массив, который указан в файле
        # choose_parameters.py для выбранной при инициализации задачи (task)
        if self.metrics:
            parameters_to_compile.append(self.metrics)
        else:
            parameters_to_compile.append(chosen_task[2])

        # Из документации:
        # Коллбэки – это набор функций, которые запускаются при определенном 
        # этапе тренировки модели. Их можно вызывать, чтобы взглянуть на
        # внутренние состояния и статистику обучения модели. В метод fit() 
        # передаётся массив коллбэков.

        # То есть, коллбэки Keras принимает только в виде массива
        # self. – чтобы иметь доступ к этой переменной вне класса
        self.used_callbacks = []

        # На тот случай, если вдруг понадобится поддержка EarlyStopping,
        # код честно нашел где-то на гитхабе в обсуждениях.
        if self.early_stopping == True:
            early_stopping_callback = EarlyStopping(monitor="value_loss")
            used_callbacks.append(early_stopping_callback)

        # Нужно для большей совместимости с прежним кодом
        if self.batch_size == "auto":
            self.batch_size = 200

        # Компилируем модель aka подготавливаем её к тренировке.
        model.compile(loss=parameters_to_compile[0],
                      optimizer=chosen_optimizer,
                      metrics=parameters_to_compile[1])
        
        print("\nCreated a clean untrained model with the following stats:\n")
        model.summary()

        return model

    def fit(self, x_train, y_train):
        # TODO: Delete this method
        pass

# Regarding pickling of the model:
# 1. https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
#    "...It is not recommended to use pickle or cPickle to save a Keras model."
#
# 2. https://github.com/keras-team/keras/issues/789
#    "...this is not recommended and this is not guaranteed to work all the time"
#
#    В Model() нет протокольных функций консервирования __reduce_ex__ 
#    и __setstate__
#
# 3. http://zachmoshe.com/2017/04/03/pickling-keras-models.html
#    Костыль для разрешения консервирования
#
# Regarding partial fit
# 1. https://github.com/keras-team/keras/issues/4446
#    "...successive calls to fit will incrementally train the model"