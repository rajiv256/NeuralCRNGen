# Create a dataset
import os
N = 100
import math
import random
dataset = []

if not os.path.exists('data/xor'):
    os.makedirs('data/xor')

while len(dataset) <= N//2:
    x1 = random.random()
    x2 = random.random()
    x1b = math.floor(x1 + 0.5)
    x2b = math.floor(x2 + 0.5)
    y = x1b^x2b
    if y == 1:
        dataset.append([[x1, x2], y])

while len(dataset) <= N:
    x1 = random.random()
    x2 = random.random()
    x1b = math.floor(x1 + 0.5)
    x2b = math.floor(x2 + 0.5)
    y = x1b^x2b
    if y == 0:
        dataset.append([[x1, x2], y])

random.shuffle(dataset)


val = []
while len(val) <= N//2:
    x1 = random.random()
    x2 = random.random()
    x1b = math.floor(x1 + 0.5)
    x2b = math.floor(x2 + 0.5)
    y = x1b^x2b
    if y == 1:
        val.append([[x1, x2], y])

while len(val) <= N:
    x1 = random.random()
    x2 = random.random()
    x1b = math.floor(x1 + 0.5)
    x2b = math.floor(x2 + 0.5)
    y = x1b^x2b
    if y == 0:
        val.append([[x1, x2], y])
random.shuffle(val)

import pickle
with open('data/xor/train.pkl', 'wb') as fout:
    pickle.dump(dataset, fout)
with open('data/xor/val.pkl', 'wb') as fout:
    pickle.dump(val, fout)