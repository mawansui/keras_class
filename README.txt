- - - v. 2.0.0 - - -

Docs for this class will appear here in a while, stay tuned.

TODOs:
- get rid of "parameters_to_compile = []" statement as it is utterly useless.
- Interface to save a model under a specified name
- partial_fit() without writing model to disc

5.06.18 update: rewriting the class to be more model-centristic

Steps taken:

1. What if we create an alternative separate method that will return a clean 
   model and then call it in class' __init__() method? That way we could do
   something like the following:

   new_model = Keras_MLP(p = a, r = a, m = e, t = e, r = s)

   So initializing the class would immideately create a new model.

   UPD: it doesn't seem to work because the "return" in this separate method
   		returns not in the variable, assigned to the class, but somewhere inside
   		the class itself, somewhere in the __init__() method, where it is called
   		originally.

2. So it seems reasonable enough to create a special method for model
   initializing, so we could create a class object, and then yeild a clean model
   from it, like that:

   clf = Keras_MLP(p = a, r = a, m = e, t = e, r = s)
   new_model = clf.create_model()

   Not really that elegant, but you could do that in one line like following:
   new_model = Keras_MLP(p = a, r = a, m = e, t = e, r = s).create_model()

   = = = UPD = = =

   Turned out it is not possible to create a model if the used data's dimensions
   are not known. So it seems reasonable to do something like

   new_model = clf.create_model(x_train, y_train)

   You can read it like this: 
   "Create a new model, suitable for training on this particular data"