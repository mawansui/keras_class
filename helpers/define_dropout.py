def define_dropout(passed_do_value, layer_sizes):
	# принимает значение дропаута (строка или массив)
	# и котреж, описывающий количество и состав слоёв

	used_dropout = []
	how_many_layers = len(layer_sizes)

	if isinstance(passed_do_value, str) and passed_do_value.lower() == "auto":
	    constant_dropout_value = 0.5
	    for i in range(0, how_many_layers):
	    	used_dropout.append(constant_dropout_value)
	    # print("how many layers: {}".format(how_many_layers))
	    # used_dropout * 3 # или не надо минус 1?

	elif isinstance(passed_do_value, list) and len(passed_do_value) == 1:
		for i in range(0, how_many_layers):
			used_dropout.append(passed_do_value[0])

	elif len(passed_do_value) == how_many_layers:
		used_dropout = passed_do_value

	elif isinstance(passed_do_value, list):
		print("Execution failed: incorrect number of dropout layers.\n"
			"Layers: {}, dropouts: {} (should be {})".format(len(layer_sizes), len(passed_do_value), len(layer_sizes)))
		used_dropout = None
		raise ValueError("Incorrect number of dropout layers.\nLayers: {}, dropouts: {} (should be {})".format(len(layer_sizes), len(passed_do_value), len(layer_sizes)))

	else:
		print("Undefined value: {}".format(passed_do_value))
		used_dropout = None
		raise ValueError("Undefined value passed to define_dropout function: {}".format(passed_do_value))

	return used_dropout
