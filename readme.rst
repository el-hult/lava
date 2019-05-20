LAVA-R
++++++
Estimation of nonlinear systems with latent variable representation.

Developing
==========

Structure
---------

Dont violate the code structure!

The idea is to have one Lava class that does all optimization, solving, prediction etc.
To specify the model class, we use separate RegressorModel objects. They are objects (instead of function) to enable memory effects which seems reasonable from a optimization perspective.
Using a "memory effect" also seems smart from a prediction and API point of view.

Testing
-------
The tests are located in the `tests` subdirectory, and running :code:`python -m unittest` will discover that.
The tests are written with the python standard `unittest` library, and it comes with a test runner out of the box. :)
Look into the python documentation for options on how to run tests.

Using PyCharm simplify stuff, since it includes scheduling, file watching etc in a simple manner.

The `covareage` package needs to be installed for PyCharm to deliver covarage reports!

