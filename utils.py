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

def transfrom_batch(im):

    assert np.shape(im)[0] == 64, "This is not a mini-batch(64)"

    im_ = np.zeros((8*128,8*128))
    
    for i in range(64):
        m = i//8
        n = i%8
        im_[m*128:(m+1)*128,n*128:(n+1)*128] = im[i]

    for i in range(1,8):
        im_[i*128,:] = 255
        im_[:,i*128] = 255

    return im_
