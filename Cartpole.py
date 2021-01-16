import math
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import DQN
import setup


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# Create a tuple to store expereince replay memory
Expereince = namedtuple('Expereince',('state','action','next_state','reward'))
x
 