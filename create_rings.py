import os
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def create_rings_dataset(N, r1=0.5, r2=1.0, r3=1.5):

    dataset = []
    neg = []
    pos = []
    while len(neg) <= N//2:
        x1 = random.uniform(-0.5, 0.5)
        x2 = abs(random.uniform(-0.5, 0.5))
        if np.linalg.norm([x1, x2]) <= 0.5:
            neg.append(([x1, x2], -1.0))
    
    while len(pos) <= N/2:
        x1 = random.uniform(-1.5, 1.5)
        x2 = abs(random.uniform(-1.5, 1.5))
        if np.linalg.norm([x1, x2]) <= 1.5 and np.linalg.norm([x1, x2]) >= 1:
            pos.append(([x1, x2], 1.0))
    dataset = pos + neg
    random.shuffle(dataset)
    return dataset

train = create_rings_dataset(150)
val = create_rings_dataset(140)

# os.mkdir('data')
# os.mkdir('data/rings')

with open('data/rings/train.pkl', "wb") as f:
    pkl.dump(train, f)

with open('data/rings/val.pkl', "wb") as f:
    pkl.dump(val, f)


# Xs = [item[0] for item in dataset]
# Ys = [item[1] for item in dataset]
# xx1 = [item[0] for item in Xs]
# xx2 = [item[1] for item in Xs]
# plt.scatter(xx1, xx2, c=Ys)
# plt.savefig("pyrings.png")