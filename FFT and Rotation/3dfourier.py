
from lib import method as M
import numpy as np

import time

import argparse
parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_mrc',type=str,default='tmp_rotate.mrc')
parser.add_argument('--input_table',type=str,default='volume8.log')
parser.add_argument('--output_mrc',type=str,default='rslice.mrc')
parser.add_argument('--angpix',type=float,default=1)
parser.add_argument('--rot_seq',type=str,default='zxz')

args=parser.parse_args()


def ffts(arr):
  return np.log(np.abs(np.fft.fftshift(np.fft.fft2(arr))))

def plot_fft(arr):
  return np.log(np.abs(arr))

def dff(arr):
  return np.fft.fftshift(np.fft.fft2(arr)) 

def lowpass(res):
  Fs=1/res/Fang
  #padding zero out of this range 
  #print(Fs)
  return int(Fs)


def crop(arr,crop_size):
  x,y,z=arr.shape
  tmp=np.zeros(shape=arr.shape,dtype=arr.dtype)
  xmin=int(x/2-crop_size)
  xmax=int(x/2+crop_size)
  ymin=int(y/2-crop_size)
  ymax=int(y/2+crop_size)
  zmin=int(z/2-crop_size)
  zmax=int(z/2+crop_size)
  #print(xmin,xmax)
  tmp[xmin:xmax,ymin:ymax,zmin:zmax]=arr[xmin:xmax,ymin:ymax,zmin:zmax]
  '''
  arr[0:xmin]=0
  arr[xmin:x]=0

  arr[:,0:ymin,:]=0
  arr[:,ymin:y,:]=0

  arr[:,:,0:zmin]=0
  arr[:,:,zmin:z]=0
  '''
  #return arr[xmin:xmax,ymin:ymax,zmin:zmax]
  return tmp
 

volume,apix=M.read_pix_mrc(args.input_mrc)
x,y,z=volume.shape

Rang=x*apix['x']
Fang=1/Rang
Fre=Fang*x*1/2
Rre=1/Fre
#print(lowpass(float(2)))
fvolume=np.fft.fftn(volume)

fshift=np.fft.fftshift(fvolume)
#fcrop=fshift
#fcrop=crop(fshift,lowpass(16.0))
#print(fcrop.shape)
#fdraw=np.log(np.abs(fshift)**2)
#fdraw=(fdraw-np.mean(fdraw))/np.std(fdraw)

#rcrop=(np.fft.ifftn(np.fft.ifftshift(fcrop)))


#M.write_file(np.float32(rcrop),'lowpass.mrc')


fslice=fshift[:,80,:]
rslice= np.abs ( np.fft.ifftn( np.fft.ifftshift (fslice) )  )


M.write_file(np.float32(rslice),args.output_mrc)

