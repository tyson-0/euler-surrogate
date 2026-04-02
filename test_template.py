from core import Euler

# this basically says "sim" is a variable which now is an object of class "Euler"
# now officially "sim" can do all the stuff below.
sim = Euler(r"A:\Euler_Surrogate\test\test_fixed_bc.csv", physics_loss=False)

# just to train your model. 
# epoch is just how many loops you want to train the model.
sim.fit(epochs=8000)

# pass in ther list of inputs you want to predict
# exactly in the order you entered in the config.yaml file
sim.predict([100, 50, 0.5, 200, 50000])


# To save the model, you need to enter a path. 
# The file's name should have the extension '.pt'
sim.save(r"A:\new\new_model.pt")


# to learn how to load a trained model, refer to test_load_model.py
# if you want to save the overhead of coding, 
# use the web app, which is pretty straight forward