"""
Все доступные функции активации:
softmax, elu, selu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid, linear

Все доступные функции ошибки:
mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, 
mean_squared_logarithmic_error, squared_hinge, hinge, categorical_hinge, logcosh,
categorical_crossentropy, sparse_categorical_crossentropy, binary_crossentropy,
kullback_leibler_divergence, poisson, cosine_proximity

Все доступные метрики:
binary_accuracy, categorical_accuracy, sparse_categorical_accuracy, 
top_k_categorical_accuracy, sparse_top_k_categorical_accuracy

массив: [активация последнего слоя, функция ошибки, массив метрик]
"""

options = {
	"regression": ["relu", "mse", ["binary_accuracy"]],
	"classification": ["sigmoid", "categorical_crossentropy", ["binary_accuracy"]],
	"multiclass": ["softmax", "categorical_crossentropy", ["binary_accuracy"]],
	"default": ["relu", "mse", ["binary_accuracy"]]
}