all_available_activation_functions = ["softmax", "elu", "selu", "softplus", 
									  "softsign", "relu", "tanh", "sigmoid", 
									  "hard_sigmoid", "linear"]

def define_activations(passed_activation, passed_layer_sizes):
	# Принимает заданное пользователем значение функции активации
	# и количество слоёв в сети.
	# Возвращает массив функций активаций.

	used_activations = []

	if isinstance(passed_activation, str) and passed_activation.lower() == "auto":
		# автоматическое присвоение ReLU
		for i in range(0, len(passed_layer_sizes)):
			used_activations.append("relu") # - - - - c - u - s - t - o - m - - 
	elif isinstance(passed_activation, str) and passed_activation.lower() in all_available_activation_functions:
		# автоматическое присвоение переданной функции активации
		for i in range(0, len(passed_layer_sizes)):
			used_activations.append(passed_activation.lower())
	elif isinstance(passed_activation, list):
		# проверка – сколько элементов в списке?
		if len(passed_activation) == (len(passed_layer_sizes)):
			for activation_function in passed_activation:
				if activation_function not in all_available_activation_functions:
					raise ValueError("Passed activation functions list contains an unknown value. Typo possible.")
			# если функций активаций столько же, сколько слоёв,
			# и в этом массиве содержатся доступные в keras функции активации,
			# то вернуть этот же массив
			return passed_activation
		else:
			raise ValueError("The number of elements in activation functions array does not equal the passed number of hidden layers")
	return used_activations