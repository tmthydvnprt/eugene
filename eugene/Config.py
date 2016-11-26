"""
Config.py - Package wide variables.

Customize with user configuration after importing
```
import eugene.Config

eugene.Config.VAR = {
    'x' : 2,
    'y' : -34,
}
eugene.Config.TRUTH = np.abs(eugene.Config.VAR['y']) + eugene.Config.VAR['x']

```
"""

import numpy as np

VAR = {
    'x' : np.arange(10)
}
TRUTH = np.arange(10)
