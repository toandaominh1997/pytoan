import numpy as np 
import matplotlib.pyplot as plt 
import math

def imshow(*kwarg, title=None, ncols=2, figsize=(10, 10)):
    num_image = len(kwarg)
    f, ax = plt.subplots(math.ceil(num_image/2.0), ncols, figsize=(10, 10))
    print(ax.shape)
    for idx, img in enumerate(kwarg):
        img = np.array(img, dtype = np.uint8)
        if num_image <= ncols:
            ax[int(idx%2)].imshow(img)
        else:
            ax[int(idx/2), int(idx%2)].imshow(img)
        if title is not None:
            try:
                if num_image <= ncols:
                    ax[int(idx%2)].set_title(title[idx])
                else:
                    ax[int(idx/2), int(idx%2)].set_title(title[idx])
            except IndexError:
                if num_image <= ncols:
                    ax[int(idx%2)].set_title('Figure: {}'.format(idx))
                else:
                    ax[int(idx/2), int(idx%2)].set_title('Figure {}'.format(idx))