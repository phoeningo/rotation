import math
import numpy as np
from lib import method as M
from numba import cuda,jit
import time
Pi=math.pi

import copy

import argparse
parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_mrc',type=str,default='0mrc.mrc')
parser.add_argument('--output_mrc',type=str,default='tmp_slice.mrc')
parser.add_argument('--angpix',type=float,default=1)
parser.add_argument('--rot_seq',type=str,default='zxz')

args=parser.parse_args()


def Rx(input_angle):
  theta=input_angle
  TT=np.matrix([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
  #print(TT)
  return TT

def Ry(input_angle):
  phi=input_angle
  TT=np.matrix([ [math.cos(phi),0,math.sin(phi)] , [0,1,0] , [-math.sin(phi),0,math.cos(phi)] ] )
 # print( TT)
  return TT


def Rz(input_angle):
  psi=input_angle
  TT= np.matrix([[math.cos(psi),-math.sin(psi),0],[math.sin(psi),math.cos(psi),0],[0,0,1]])
  #print (TT)
  return TT


  
def euler2matrix(input_angle,rotate_seq):
  #phi,theta,psi=input_angle
  #angles=[]  
  Rs=[]
  for ri in range(3):
    if rotate_seq[ri]=='x':
      # Here T changed. Why?
      Rs.append(Rx(input_angle[ri]))
    if rotate_seq[ri]=='y':
      Rs.append(Ry(input_angle[ri]).T)
    if rotate_seq[ri]=='z':
      Rs.append(Rz(input_angle[ri]).T)

  return np.array((Rs[0]*Rs[1]*Rs[2]).T)



@cuda.jit
def calc(volume,R,out_volume,input_matrix,output_matrix):
  #output_volume=np.zeros(shape=volume.shape,dtype=volume.dtype)
  x,y,z=volume.shape
  radius=int(x/2)
  i,j,k=cuda.grid(3)
  # input was right-hand 
  xin=k-radius
  yin=j-radius

  #zin=radius-i
  zin=i-radius
  # convert to left-hand ,then multiply with R
  
  for mj in range(3):
    output_matrix[0,mj]=xin*R[0,mj]+yin*R[1,mj]+zin*R[2,mj]

  #inter
  fi=output_matrix[0,0]
  fj=output_matrix[0,1]
  fk=output_matrix[0,2]

  if abs(fi)<radius and abs(fj)<radius and abs(fk)<radius:
    oi=int(fi)
    oj=int(fj)
    ok=int(fk)

    delta_i=fi-oi
    delta_j=fj-oj
    delta_k=fk-ok
    # when you read from volume, you should use right-hand 
    # and convert back ,to read from volume
  
    oi+=radius
    oj+=radius
    ok+=radius
    
    tmp=oi
    oi=ok
    ok=tmp
    #ok=2*radius-tmp
    
    tmp_vi=(delta_i)*(volume[oi+1,oj,ok]-volume[oi,oj,ok])
    tmp_vj=(delta_j)*(volume[oi,oj+1,ok]-volume[oi,oj,ok])
    tmp_vk=(delta_k)*(volume[oi,oj,ok+1]-volume[oi,oj,ok])
    tmp_value=volume[oi,oj,ok]+tmp_vi+tmp_vj+tmp_vk

    out_volume[i,j,k]=tmp_value


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



R=euler2matrix((45,45,10),'zxz')
#R=np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
center=np.int32((0,0,0))
input_map_A,apix=M.read_pix_mrc(args.input_mrc)


output_map=np.zeros(shape=input_map_A.shape,dtype=input_map_A.dtype)
blockdim=input_map_A.shape
vector3=np.zeros([1,3],dtype=np.float32)
o_vec=np.zeros([1,3],dtype=np.float32)

calc[blockdim,1](input_map_A,R,output_map,vector3,o_vec)

#print(np.max(output_map))
volume=output_map
x,y,z=volume.shape

Rang=x*apix['x']
Fang=1/Rang
Fre=Fang*x*1/2
Rre=1/Fre
fvolume=np.fft.fftn(volume)

fshift=np.fft.fftshift(fvolume)
fslice=fshift[:,127,:]
rslice= np.abs ( np.fft.ifftn( np.fft.ifftshift (fslice) )  )


M.write_file(np.float32(rslice),args.output_mrc)





