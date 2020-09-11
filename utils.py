import numpy as np
from PIL import Image
import colorsys

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

def convert_bw_im2hsv_im(bw,color):

    if np.max(bw)>1:
        bw = bw/255.

    r = color[0]
    g = color[1]
    b = color[2]
    hsv = colorsys.rgb_to_hsv(r, g, b)
    h = hsv[0]
    s = hsv[1]
    new_im = np.zeros((np.shape(bw)[0], np.shape(bw)[0], 3))

    for i in range(np.shape(bw)[0]):
        for j in range(np.shape(bw)[1]):
            if bw[i,j]>0:
                ori_hsv = colorsys.rgb_to_hsv(bw[i,j], bw[i,j], bw[i,j]*0.8)
                new_hsv = (h, s, ori_hsv[2])
                new_rgb = colorsys.hsv_to_rgb(new_hsv[0], new_hsv[1], new_hsv[2])
                new_im[i,j] = [255*new_rgb[0], 255*new_rgb[1], 255*new_rgb[2]]
      
    return new_im

def overlap_im(im1,im2):

    if np.max(im1)>1:
        im1 = im1/255.

    if np.max(im2)>1:
        im2 = im2/255.

    yellow_rgb = (0, 1., 1.) #bgr
    yellow_hsv = colorsys.rgb_to_hsv(yellow_rgb[0], yellow_rgb[1],yellow_rgb[2])

    orange_rgb = (25/255.,133/255.,240/255.)
    orange_hsv = colorsys.rgb_to_hsv(orange_rgb[0], orange_rgb[1],orange_rgb[2])

    blue_rgb = (233/255.,162/255.,0)
    blue_hsv = colorsys.rgb_to_hsv(blue_rgb[0], blue_rgb[1],blue_rgb[2])

    new_im = np.zeros((128,128,3))
      
    for m in range(128):
        for n in range(128):
            if im1[m,n] > 0 and im2[m,n] > 0:
                new_im[m,n] = [255*yellow_rgb[0], 255*yellow_rgb[1],255*yellow_rgb[2]]
            elif im1[m,n] > 0 and im2[m,n] == 0:
                ori_hsv = colorsys.rgb_to_hsv(im1[m,n], im1[m,n], im1[m,n])
                new_hsv = (blue_hsv[0], blue_hsv[1], ori_hsv[2])
                new_rgb = colorsys.hsv_to_rgb(new_hsv[0], new_hsv[1], new_hsv[2])    
                new_im[m,n] = [255*new_rgb[0], 255*new_rgb[1], 255*new_rgb[2]]
            else:
                ori_hsv = colorsys.rgb_to_hsv(im2[m,n], im2[m,n], im2[m,n])
                new_hsv = (orange_hsv[0], orange_hsv[1], ori_hsv[2])
                new_rgb = colorsys.hsv_to_rgb(new_hsv[0], new_hsv[1], new_hsv[2])    
                new_im[m,n] = [255*new_rgb[0], 255*new_rgb[1], 255*new_rgb[2]]

    return new_im

def overlap_batch(im1,im2):

    if np.max(im1)>1:
        im1 = im1/255.

    if np.max(im2)>1:
        im2 = im2/255.

    new_im = np.zeros((128*8,128*8*3+128,4))

    for i in range(64):

        im_a = im1[i]
        im_b = im2[i]

        im_c = overlap_im(im_a,im_b)

        m = i//8
        n = i%8

        new_im[m*128:(m+1)*128,n*128:(n+1)*128,0] = im_a*255
        new_im[m*128:(m+1)*128,n*128:(n+1)*128,1] = im_a*255
        new_im[m*128:(m+1)*128,n*128:(n+1)*128,2] = im_a*255
        new_im[m*128:(m+1)*128,n*128:(n+1)*128,3] = 255

        new_im[m*128:(m+1)*128,64+128*8+n*128:64+128*8+(n+1)*128,0] = im_b*255
        new_im[m*128:(m+1)*128,64+128*8+n*128:64+128*8+(n+1)*128,1] = im_b*255
        new_im[m*128:(m+1)*128,64+128*8+n*128:64+128*8+(n+1)*128,2] = im_b*255
        new_im[m*128:(m+1)*128,64+128*8+n*128:64+128*8+(n+1)*128,3] = 255

        new_im[m*128:(m+1)*128,128+128*16+n*128:128+128*16+(n+1)*128,:3] = im_c
        new_im[m*128:(m+1)*128,128+128*16+n*128:128+128*16+(n+1)*128,3] = 255

    
    yellow_rgb = (0, 1., 1.) #bgr
    yellow_hsv = colorsys.rgb_to_hsv(yellow_rgb[0], yellow_rgb[1],yellow_rgb[2])

    orange_rgb = (25/255.,133/255.,240/255.)
    orange_hsv = colorsys.rgb_to_hsv(orange_rgb[0], orange_rgb[1],orange_rgb[2])

    blue_rgb = (233/255.,162/255.,0)
    blue_hsv = colorsys.rgb_to_hsv(blue_rgb[0], blue_rgb[1],blue_rgb[2])
    

    for i in range(1,8):
        new_im[i*128,:128*8,:3] = [255*blue_rgb[0], 255*blue_rgb[1],255*blue_rgb[2]]
        new_im[:,i*128,:3] = [255*blue_rgb[0], 255*blue_rgb[1],255*blue_rgb[2]]

    for i in range(1,8):
        new_im[i*128,64+128*8:64+128*16,:3] = [255*orange_rgb[0], 255*orange_rgb[1],255*orange_rgb[2]]
        new_im[:,64+128*8+i*128,:3] = [255*orange_rgb[0], 255*orange_rgb[1],255*orange_rgb[2]]

    for i in range(1,8):
        new_im[i*128,128+128*16:128+128*24,:3] = [255, 255, 255]
        new_im[:,128+128*16+i*128,:3] = [255, 255, 255]

    # new_im[:,128*8,:] = [255, 255, 255]
    # new_im[:,128*16,:] = [255, 255, 255]

    return new_im


def overlap_regression(im1,im2):

    if np.max(im1)>1:
        im1 = im1/255.

    if np.max(im2)>1:
        im2 = im2/255.

    new_im = np.zeros((128*8,128*8*3+128,4))

    im_a = im1
    im_a_large = cv2.resize(im_a,(128*4,128*4))
    new_im[2*128:6*128,2*128:6*128,0] = im_a_large*255
    new_im[2*128:6*128,2*128:6*128,1] = im_a_large*255
    new_im[2*128:6*128,2*128:6*128,2] = im_a_large*255
    new_im[2*128:6*128,2*128:6*128,3] = 255
    
    new_im[2*128:6*128,2*128,:3] = [255*blue_rgb[0], 255*blue_rgb[1],255*blue_rgb[2]]
    new_im[2*128:6*128,6*128,:3] = [255*blue_rgb[0], 255*blue_rgb[1],255*blue_rgb[2]]
    new_im[2*128,2*128:6*128,:3] = [255*blue_rgb[0], 255*blue_rgb[1],255*blue_rgb[2]]
    new_im[6*128,2*128:6*128,:3] = [255*blue_rgb[0], 255*blue_rgb[1],255*blue_rgb[2]]


    for i in range(64):

        im_b = im2[i]

        im_c = overlap_im(im_a,im_b)

        m = i//8
        n = i%8

        

        new_im[m*128:(m+1)*128,64+128*8+n*128:64+128*8+(n+1)*128,0] = im_b*255
        new_im[m*128:(m+1)*128,64+128*8+n*128:64+128*8+(n+1)*128,1] = im_b*255
        new_im[m*128:(m+1)*128,64+128*8+n*128:64+128*8+(n+1)*128,2] = im_b*255
        new_im[m*128:(m+1)*128,64+128*8+n*128:64+128*8+(n+1)*128,3] = 255

        new_im[m*128:(m+1)*128,128+128*16+n*128:128+128*16+(n+1)*128,:3] = im_c
        new_im[m*128:(m+1)*128,128+128*16+n*128:128+128*16+(n+1)*128,3] = 255

    
    yellow_rgb = (0, 1., 1.) #bgr
    yellow_hsv = colorsys.rgb_to_hsv(yellow_rgb[0], yellow_rgb[1],yellow_rgb[2])

    orange_rgb = (25/255.,133/255.,240/255.)
    orange_hsv = colorsys.rgb_to_hsv(orange_rgb[0], orange_rgb[1],orange_rgb[2])

    blue_rgb = (233/255.,162/255.,0)
    blue_hsv = colorsys.rgb_to_hsv(blue_rgb[0], blue_rgb[1],blue_rgb[2])

    for i in range(1,8):
        new_im[i*128,64+128*8:64+128*16,:3] = [255*orange_rgb[0], 255*orange_rgb[1],255*orange_rgb[2]]
        new_im[:,64+128*8+i*128,:3] = [255*orange_rgb[0], 255*orange_rgb[1],255*orange_rgb[2]]

    for i in range(1,8):
        new_im[i*128,128+128*16:128+128*24,:3] = [255, 255, 255]
        new_im[:,128+128*16+i*128,:3] = [255, 255, 255]

    return new_im