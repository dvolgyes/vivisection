Vivisection
===========

A module for debugging machine learning models.
At this moment it supports PyTorch, mostly, and some general tools.


Installation
------------

You can install the code with *pip* using this shortcut:
```make install```

or manually with the setup.py file.

Removing the package:
```make uninstall```


Usage
-----

```
#your imports
import necessary_modules

import vivisection
from vivisection import SampleLogger, debug_model
...

#optional, only for debugging data samples
dataloader = DataLoader(...., SampleLogger(Sampler(...))
...

# create your model in any way you want
model = create_model()

model = debug_model(model)

for i in range(epochs):
    ...
    loss = lossfunction(prediction, groundt)

    # enable create graph for more detailed info
    # if the error originates from the loss function
    loss.backward(create_graph=True)
    ...
    optimizer.step()
    ...
```

Main options:

 - you can enable/disable the forward/backward/forward_pre hooks
 - you can set manually design test functions (default: isnan/isfinite)
 - log can be redirected to file (not yet implemented)
 - sample index inside dataset is look up, if applicable
 - abort on error (default), or fix (nan->0, +/-inf -> fixed number)

TODO
----

More documentation.
