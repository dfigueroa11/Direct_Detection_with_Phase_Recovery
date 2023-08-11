import torch as t
import numpy as np
from itertools import product



mapping = t.tensor([2,3,5], dtype=t.cfloat)
channel = t.tensor([7,11,13])
memory = len(channel)-1


tx_spaces_early = [t.tensordot(t.tensor(list(product(mapping, repeat=l+1))),t.flip(channel[:l+1].cfloat(), dims=[-1, ]),dims=[[-1, ], [0, ]]) for l in range(memory)]
tx_space = t.tensordot(t.tensor(list(product(mapping, repeat=memory+1))),t.flip(channel[:memory+1].cfloat(), dims=[-1, ]),dims=[[-1, ], [0, ]])
tx_spaces_late = [t.tensordot(t.tensor(list(product(mapping, repeat=l+1))),t.flip(channel[-(l+1):].cfloat(), dims=[-1, ]),dims=[[-1, ], [0, ]]) for l in range(memory)]

print(tx_spaces_early[0])
print(tx_spaces_early[0].unsqueeze(0))


