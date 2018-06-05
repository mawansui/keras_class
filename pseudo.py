clf = Keras_MLP(p = a,
				r = a,
				m = e,
				t = e,
				r = s)

model = clf.create_model() # создаёт модель
model.fit(x_data, y_data) # обучает модель

# по моей логике, этот код будет дообучать модель, которая болтается как
# переменная в оперативной памяти
for x_chunk, y_chunk in all_data:
	model.fit(x_chunk, y_chunk)

# а ещё в керасе есть вот такая штука:
for x_chunk, y_chunk in all_data:
	model.train_on_batch(x_chunk, y_chunk)

# Тоже, по сути, дообучение модели, которая валяется в оперативе.

# Надо проверить, можно ли обойтись одним только методом fit()
# Согласно этому треду (https://github.com/keras-team/keras/issues/4446), 
# << successive calls to fit will incrementally train the model >>

# Значит, надо скопировать (резервно) то, что уже есть, и переписать класс
# так, чтобы он был центрирован на МОДЕЛИ.

