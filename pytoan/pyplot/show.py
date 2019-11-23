import numpy as np 
import matplotlib.pyplot as plt 
import math

def imshow(*kwarg, title=None, ncols=2, figsize=(10, 10)):
    num_image = len(kwarg)
    if ncols<1:
        print('WARNING: Columns must greater than 0')
        return None
    if num_image < 1:
        print('WARNING: Number of image must greater than 0')
    
    if ncols ==1 and num_image==1:
        plt.figure()
        plt.imshow(kwarg[0])
        return True 
    if ncols==1 or num_image==1:
        f, ax = plt.subplots(max(ncols, num_image), figsize=figsize)
    else:
        f, ax = plt.subplots(math.ceil(num_image/2.0), ncols, figsize=figsize)
    for idx, img in enumerate(kwarg):
        img = np.array(img, dtype = np.uint8)
        if ncols==1 or num_image==1:
            ax[int(idx%2)].imshow(img)
        else:
            ax[int(idx/2), int(idx%2)].imshow(img)
        if title is not None:
            try:
                if ncols==1 or num_image==1:
                    ax[int(idx%2)].set_title(title[idx])
                else:
                    ax[int(idx/2), int(idx%2)].set_title(title[idx])
            except IndexError:
                if ncols==1 or num_image==1:
                    ax[int(idx%2)].set_title('Figure: {}'.format(idx))
                else:
                    ax[int(idx/2), int(idx%2)].set_title('Figure {}'.format(idx))