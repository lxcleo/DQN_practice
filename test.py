import setup as st
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import gym
import numpy as np
'''
device = torch.device("cuda" if to rch.cuda.is_available() else"cpu")
em = st.EnvManager(device)
em.reset()
screen = em.render('rgb_array')
screen = em.get_processed_screen()plt.figure()
plt.imshow(screen.squeeze(0).permute(1, 2, 0), interpolation = 'none')
plt.show()

'''

st.plot(np.random.rand(300), 100)
