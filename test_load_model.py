from core import Euler

# paste the location of the model you want to load into the arg of from_saved() method
saved_sim = Euler.from_saved(r"A:\new\new_model.pt")

# start predicting and using the model.
saved_sim.predict([100, 50, 1, 2, 3])