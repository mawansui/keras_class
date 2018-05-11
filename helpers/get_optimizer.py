from keras import optimizers

def get_optimizer(optimizer_name, kwargs):
	# Принимает строку и словарь с параметрами для оптимизатора
	# По соответствию с этими данными возвращает оптимизатор с заданными пар-ми
	# Если оптимизатора с таким названием нет, выводит сообщ. и возвр. дефолтный

	name = optimizer_name.lower()

	if name == "adam":
		return optimizers.Adam(lr = kwargs.get('lr', 0.001),
							   beta_1 = kwargs.get('beta_1', 0.9),
							   beta_2 = kwargs.get('beta_2', 0.999),
							   epsilon = kwargs.get('epsilon', 0),
							   decay = kwargs.get('decay', 0.0))
	
	elif name == "sgd":
		return optimizers.SGD(lr = kwargs.get('lr', 0.01),
							  momentum = kwargs.get('momentum', 0.0),
							  decay = kwargs.get('decay', 0.0),
							  nesterov = kwargs.get('nesterov', False))

	elif name == "rmsprop":
		return optimizers.RMSprop(lr = kwargs.get('lr', 0.001),
								  rho = kwargs.get('rho', 0.9),
								  epsilon = kwargs.get('epsilon', 0),
								  decay = kwargs.get('decay', 0.0))
	elif name == "adagard":
		return optimizers.Adagard(lr=kwargs.get('lr', 0.01), 
								  epsilon=kwargs.get('epsilon', 0), 
								  decay=kwargs.get('decay', 0.0))

	elif name == "adadelta":
		return optimizers.Adadelta(lr=kwargs.get('lr', 1.0), 
								   rho=kwargs.get('rho', 0.95), 
								   epsilon=kwargs.get('epsilon', 0), 
								   decay=kwargs.get('decay', 0.0))

	elif name == "adamax":
		return optimizers.Adamax(lr = kwargs.get('lr', 0.002), 
								 beta_1=kwargs.get('beta_1', 0.9), 
								 beta_2=kwargs.get('beta_2', 0.999), 
								 epsilon=kwargs.get('epsilon', 0), 
								 decay=kwargs.get('decay',0.0))
	elif name == "nadam":
		return optimizers.Nadam(lr=kwargs.get('lr', 0.002), 
							 	beta_1=kwargs.get('beta_1', 0.9), 
							 	beta_2=kwargs.get('beta_2', 0.999), 
							 	epsilon=kwargs.get('epsilon', 0), 
							 	schedule_decay=kwargs.get('schedule_decay',0.004))

	else:
		print("No such optimizer ({}) in the Keras Library!"
			  " Returning default Adam.".format(name))
		return optimizers.Adam(lr = 0.001,
							   beta_1 = 0.9,
							   beta_2 = 0.999,
							   epsilon = 0,
							   decay = 0.0)