#!/usr/bin/env python
# coding: utf-8

# In[9]:


# -*- coding: utf-8 -*-
"""MA4268-Project1
"""
from PIL import Image as image
import numpy as np
import math,cmath
import matplotlib.pyplot as plt
import time
import argparse

def dct(len):
  import numpy as np
  N = len
  lamb = np.ones(N, np.float64)
  lamb[0] = 1/(2**0.5)
  lamb = lamb*(2**0.5)/(N**0.5)
  coef_mat = np.empty((N,N), np.float64)
  for i in range(N):
    for j in range(N):
      coef_mat[i,j] = lamb[i]*math.cos(i*math.pi/N*(j+1/2))
  return coef_mat

def dct2d(im):
# Computing 2D Discrete_Cosine_Transform(DCT) of a given ndarray im.
# Parameters: 
#   im: ndarray.    An array representing image.
# Returns:
#   out: ndarray.   An array representing 2D DCT coefficients.
  
  N = im.shape[0]
  M = im.shape[1]
  if N == M:
    m = dct(N)
    out = np.matmul(np.matmul(m, im), m.T)
  else:
    out = np.matmul(np.matmul(dct(N), im), dct(M).T)
  
  return out

def idct(len):
  N = len

  lamb = np.ones(N, np.float64)
  lamb[0] = 1/(2**0.5)
  lamb = lamb*(2**0.5)/(N**0.5)

  icoef_mat = np.empty((N,N), np.float64)
  for i in range(N):
    for j in range(N):
      icoef_mat[i,j] = lamb[j]*math.cos(j*math.pi/N*(i+1/2))
  
  return icoef_mat

def idct2d(coef):
# Computing an image in the form of ndarray from the ndarray coef which represents its DCT coefficients.
# Parameters: 
#   coef: ndarray.   An array representing 2D DCT coefficients.
# Returns:
#   out: ndarray.   An array representing the image reconstructed from its 2D DCT coefficients.

  #IDCT coefficient matrix
  N = coef.shape[0]
  M = coef.shape[1]
  if N == M :
    m = idct(N)
    out = np.matmul(np.matmul(m, coef), m.T)
  else:
    out = np.matmul(np.matmul(idct(N), coef), idct(M).T)
  return out

def fft(f):
# Computing 1D Fourier_Fast_Transform(FFT) of a given ndarray 1D-signal.
# Parameters: 
#   f: ndarray.    An array representing 1D-signal.
# Returns:
#   out: ndarray.   An array representing 1D FFT coefficients
  if len(f) == 1:
    return f  
  else:
    N = len(f)
    N2 = int(N/2)
    f_even = np.empty(N2, dtype=complex)
    f_odd = np.empty(N2, dtype=complex)
    tp = np.empty(N2, dtype=complex)
    F = np.empty(N, dtype=complex)
    # print(len(f_even),len(f_odd),len(tp),len(F))
    for n in range(N2):
      tp[n] = cmath.exp(-cmath.sqrt(-1)*2*math.pi*n/N)
    f_even = f[0:N2]+f[N2:N] 
    f_odd = (f[0:N2]-f[N2:N])*tp
    F[1::2], F[::2] = fft(f_even), fft(f_odd)

  return F

def fft2d(im):
# Computing 2D Fourier_Fast_Transform(FFT) of a given ndarray im.
# Parameters: 
#   im: ndarray.    An array representing image.
# Returns:
#   out: ndarray.   An array representing 2D FFT coefficients
  N = im.shape[0]
  M = im.shape[1]
  im = np.array(im,dtype=complex)
  for i in range(N):
    im[i,:] = fft(im[i,:])
  for j in range(M):
    im[:,j] = fft(im[:,j])
  out = im
  return out

def ifft(coef):
# Computing an 1D-signal in the form of ndarray from the ndarray coef which represents its FFT coefficients.
# Parameters: 
#   coef: ndarray.   An array representing 1D FFT coefficients.
# Returns:
#   out: ndarray.   An array representing the image reconstructed from its 1D FFT coefficients.
  N = len(coef)
  #Adjust the position, (f^[k])*=f[N-k]
  out = np.roll(fft(np.flip(coef)/N).conjugate(),1)
  return out

def ifft2d(coef):
# Computing an image in the form of ndarray from the ndarray coef which represents its FFT coefficients.
# Parameters: 
#   coef: ndarray.   An array representing 2D FFT coefficients.
# Returns:
#   out: ndarray.   An array representing the image reconstructed from its 2D FFT coefficients.
  N = coef.shape[0]
  M = coef.shape[1]

  coef2 = coef.copy()
  #Implement ifft on rows and columns
  for i in range(N):
    coef2[i,:] = ifft(coef2[i,:])
  for j in range(M):
    coef2[:,j] = ifft(coef2[:,j])

  coef3 = coef2.copy()
  # # Adjust the position
  for i in range(1,N):
    coef3[i,:] = coef2[N-i,:]
  coef3 = np.real(coef3)
  out = coef3

  return out

def Np2(n):
#To find the nearest N of 2^N greater than n
  a = 0
  while n>2**a: 
    a += 1
  return a

def zero_filling(im):
#To let the prictures be a 2^N * 2^N pictures
  N = im.shape[0]
  M = im.shape[1]
  if N != 2**Np2(N):
    pad = 2**Np2(N)-N
    im = np.pad(im,((0,pad),(0,0)),"constant")
    if M != 2**Np2(M):
      pad = 2**Np2(M)-M
      im = np.pad(im,((0,0),(0,pad)),"constant")
  return im

def error_check(im,img):
#To get the L2-norm of round error
#im: the recovery matrix
#img: the original matrix
  error = 0
  for i in range(im.shape[0]):
    for j in range(im.shape[1]):
      if (img[i, j] != im[i, j]):
        error += (img[i, j] - im[i, j]) ** 2
  return np.sqrt(error)

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="project")
  parser.add_argument("--img_path", type=str, default='./pepper_gray.png', help='The test image path')
  parser.add_argument("--save_pth1", type=str, default='./dct2d_recovery.png', help="The save path of reconstructed image1")
  parser.add_argument("--save_pth2", type=str, default='./fft2d_recovery.png', help="The save path of reconstructed image2 ")
  opt = parser.parse_args()

  img_path = opt.img_path  # The test image path
  img = np.array(image.open(img_path).convert('L'))

  # dct2d_coef = dct2d(img)
  # fig_dct2d = idct2d(dct2d_coef)
  # recovery = np.uint8(np.interp(fig_dct2d, (fig_dct2d.min(), fig_dct2d.max()), (0, 255)))
  # recovery = image.fromarray(recovery, mode='L')
  # save_pth = opt.save_pth1
  # recovery.save(save_pth)
  # np.save('./dct2d_coeff.npy', dct2d_coef)

  #codes to process NXM size pictures
  fft2d_coef = fft2d(zero_filling(img))
  fig_fft2d = ifft2d(fft2d_coef)[:img.shape[0],:img.shape[1]]

  #codes to process NXN size pictures
  # fft2d_coef = fft2d(img)
  # fig_fft2d = ifft2d(fft2d_coef)
  # recovery = np.uint8(np.interp(fig_fft2d, (fig_fft2d.min(), fig_fft2d.max()), (0, 255)))
  # recovery = image.fromarray(recovery, mode='L')
  # save_pth = opt.save_pth2
  # recovery.save(save_pth)
  # np.save('./fft2d_coeff.npy', fft2d_coef)

  # print("DCT2d round error checking: %.10f"%error_check(fig_dct2d, img))
  # print("FFT2d round error checking: %.10f"%(error_check(fig_fft2d, img)))