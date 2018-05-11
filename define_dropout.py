def define_dropout(passed_do_value, layer_sizes):
	# принимает значение дропаута (строка или массив)
	# и котреж, описывающий количество и состав слоёв

	used_dropout = []
	how_many_layers = len(layer_sizes)

	if passed_do_value == "Auto":
	    constant_dropout_value = 0.5
	    for i in range(0, how_many_layers):
	    	used_dropout.append(constant_dropout_value)
	    print("how many layers: {}".format(how_many_layers))
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

	else:
		print("Undefined value: {}".format(passed_do_value))
		used_dropout = None

	return used_dropout


used = define_dropout("Auto", (10, 10, 10, ))
print("used values:")
print(used)
