LAVA-R
++++++
Estimation of nonlinear systems with latent variable representation.

Developing
==========

Structure
---------
Don't violate the code structure!

The idea is to have one Lava class that does all optimization, solving, prediction etc.
To specify the model class, we use separate RegressorModel objects. They are objects (instead of function) to enable memory effects which seems reasonable from a optimization perspective.
Using a "memory effect" also seems smart from a prediction and API point of view.

Testing
-------
The tests are located in the `tests` subdirectory, and running :code:`python -m unittest` will discover that.
The tests are written with the python standard `unittest` library, and it comes with a test runner out of the box. :)
Look into the python documentation for options on how to run tests.

Using PyCharm simplify stuff, since it includes scheduling, file watching etc in a simple manner.

Test coverage
.............


The :code:`covareage` package needs to be installed for PyCharm to deliver covarage reports! Try running

.. code-block::

    coverage run -m unittest discover
    coverage html

then look into the folder :code:`htmlcov` and read your coverage report.

This can be simplified quite substantially by using PyCharm, which has built in support for :code:`coverage`




Linting
-------
I try to follow all default settings in PyLint.
Some I skip though, such as implementing all abstract methods on a mocking-class.
In such cases, I use directives like :code:`# noinspection PyAbstractClass`.
Thats why such statements can be found around...
