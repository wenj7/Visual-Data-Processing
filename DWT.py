import numpy as np
import scipy
from scipy import signal
from PIL import Image
import argparse
import matplotlib.pyplot as plt


def haar2d(im,lvl):
# Computing 2D discrete Haar wavelet transform of a given ndarray im.
# Parameters: 
#   im: ndarray.    An array representing image
#   lvl: integer.   An integer representing the level of wavelet decomposition
#  Returns:
#   out: ndarray.   An array representing Haar wavelet coefficients with lvl level. It has the same shape as im

# ----

    H_rev = np.ones([2, 2], dtype=float) / 2
    G1_rev = np.array([[-1, -1], [1, 1]]) / 2
    G2_rev = np.array([[-1, 1], [-1, 1]]) / 2
    G3_rev = np.array([[1, -1], [-1, 1]]) / 2
    # zero padding
    # im = zero_padding(im, lvl)
    N = im.shape[0]
    M = im.shape[1]

    coef = np.empty([N, M])
    for i in range(1, lvl):
        div = 2 ** i
        coef[:int(N / div), :int(M / div)] = signal.convolve2d(im, H_rev, mode='same')[1::2, 1::2]
        coef[:int(N / div), int(M / div):int(2 * M / div)] = signal.convolve2d(im, G1_rev, mode='same')[1::2, 1::2]
        coef[int(N / div):int(2 * N / div), :int(M / div)] = signal.convolve2d(im, G2_rev, mode='same')[1::2, 1::2]
        coef[int(N / div):int(2 * N / div), int(M / div):int(2 * M / div)] = signal.convolve2d(im, G3_rev, mode='same')[
                                                                             1::2, 1::2]
        im = coef[:int(N / div), :int(M / div)].copy()

    out = coef
# ----

    return out
 
def ihaar2d(coef,lvl):
# Computing an image in the form of ndarray from the ndarray coef which represents its DWT coefficients.
# Parameters: 
#   coef: ndarray   An array representing 2D Haar wavelet coefficients
#   lvl: integer.   An integer representing the level of wavelet decomposition
#  Returns:
#   out: ndarray.   An array representing the image reconstructed from its Haar wavelet coefficients.

# ----

    y = coef.copy()
    N = y.shape[0]
    M = y.shape[1]
    H = np.flip(np.flip(np.ones([2, 2], dtype=float) / 2, axis=1), axis=0)
    G1 = np.flip(np.flip(np.array([[-1, -1], [1, 1]]) / 2, axis=1), axis=0)
    G2 = np.flip(np.flip(np.array([[-1, 1], [-1, 1]]) / 2, axis=1), axis=0)
    G3 = np.flip(np.flip(np.array([[1, -1], [-1, 1]]) / 2, axis=1), axis=0)

    for i in reversed(range(1, lvl)):
        div = 2 ** i
        rH = y[:int(N / div), :int(M / div)].copy()
        rG1 = y[:int(N / div), int(M / div):int(2 * M / div)].copy()
        rG2 = y[int(N / div):int(2 * N / div), :int(M / div)].copy()
        rG3 = y[int(N / div):int(2 * N / div):, int(M / div):int(2 * M / div)].copy()

        # upsampling
        for k in range(0, int(N / div)):
            rH = np.insert(rH, 2 * k + 1, values=np.zeros([1, int(N / div)]), axis=0)
            rG1 = np.insert(rG1, 2 * k + 1, values=np.zeros([1, int(N / div)]), axis=0)
            rG2 = np.insert(rG2, 2 * k + 1, values=np.zeros([1, int(N / div)]), axis=0)
            rG3 = np.insert(rG3, 2 * k + 1, values=np.zeros([1, int(N / div)]), axis=0)
        for j in range(0, int(M / div)):
            rH = np.insert(rH, 2 * j + 1, values=np.zeros([1, 2 * int(N / div)]), axis=1)
            rG1 = np.insert(rG1, 2 * j + 1, values=np.zeros([1, 2 * int(N / div)]), axis=1)
            rG2 = np.insert(rG2, 2 * j + 1, values=np.zeros([1, 2 * int(N / div)]), axis=1)
            rG3 = np.insert(rG3, 2 * j + 1, values=np.zeros([1, 2 * int(N / div)]), axis=1)

        rH = signal.convolve2d(rH, H, mode='same')
        rG1 = signal.convolve2d(rG1, G1, mode='same')
        rG2 = signal.convolve2d(rG2, G2, mode='same')
        rG3 = signal.convolve2d(rG3, G3, mode='same')

        y[:int(2 * N / div), :int(2 * M / div)] = rH + rG1 + rG2 + rG3

    out = y
# ----
    return out

if __name__ == "__main__":
# Code for testing.
# Modify the img_path to the path stored image and the level of wavelet decomposition.

    import time
    parser = argparse.ArgumentParser(description="wavelet")
    parser.add_argument("--img_path",  type=str, default='./test.png',  help='The test image path')
    parser.add_argument("--level", type=int, default=4, help="The level of wavelet decomposition")
    parser.add_argument("--save_pth", type=str, default='./recovery.png', help="The save path of reconstructed image ")
    opt = parser.parse_args()

    img_path = opt.img_path # The test image path
    level = opt.level # The level of wavelet decomposition
    save_pth = opt.save_pth


    img = np.array(Image.open(img_path).convert('L'))
    # T1=time.perf_counter()

    haar2d_coef = haar2d(img,level)
    # T2=time.perf_counter()

    fig = ihaar2d(haar2d_coef,level)
    recovery = np.uint8(np.interp(fig, (fig.min(),fig.max()),(0,255)))
    error = 0
    count = 0
    for i in range(fig.shape[0]):
        for j in range(fig.shape[0]):
            if fig[i,j] != img[i,j]:
                count +=1
                error += abs(fig[i,j]-img[i,j])
    print(error)
    print(count)

    recovery =  Image.fromarray(recovery,mode='L')

    recovery.save(save_pth)
    np.save('./haar2_coeff.npy',haar2d_coef)
