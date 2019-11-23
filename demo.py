import numpy as np 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import math 
img1 = Image.open('/home/toandm2/Downloads/test.jpg')
img2 = cv2.imread('/home/toandm2/Downloads/test.jpg')


def visualization(*kwarg, title=None, ncols=2):
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

from pytoan.pyplot import imshow

imshow(img1, img1)

# plt.imshow(img)
plt.show()

# cv2.waitKey(0)          
# cv2.destroyAllWindows() 
