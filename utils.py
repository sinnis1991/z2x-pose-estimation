import numpy as np
from PIL import Image

def batch_im_save(im,name,path='/content/gdrive/My Drive/model'):

    if np.max(im)<=1:
        im = im*255
    
    im = im.astype(np.uint8)

    if np.shape(im)[-1] !=3:
        sample_im = Image.new('L', (128*8, 128*8))

        for i,x in enumerate(range(0, 128*8, 128)):
            for j,y in enumerate(range(0, 128*8, 128)):

                I1 = im[i*8+j,:,:]
                I1 = Image.fromarray(I1, mode='L')
                sample_im.paste(I1, (x, y))

        sample_im.save( os.path.join(path,'sample{}.png'.format(index)) )
