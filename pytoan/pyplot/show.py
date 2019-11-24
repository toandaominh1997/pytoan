import numpy as np
import matplotlib.pyplot as plt
import math


def imshow(*kwarg, title=None, ncols=2, figsize=(10, 10)):
    num_image = len(kwarg)
    if ncols < 1:
        print('WARNING: Columns must greater than 0. Return None')
        return None
    if num_image < 1:
        print('WARNING: Number of image must greater than 0. Return None')
        return None
    f, ax = plt.subplots(math.ceil(num_image/ncols), ncols,
                         figsize=figsize, squeeze=False)
    for idx, img in enumerate(kwarg):
        ax[int(idx/ncols), int(idx % ncols)].imshow(img)
        if title is not None:
            try:
                ax[int(idx/ncols), int(idx % ncols)].set_title(title[idx])
            except IndexError:
                ax[int(idx/ncols), int(idx % ncols)
                   ].set_title('Figure {}'.format(idx))
