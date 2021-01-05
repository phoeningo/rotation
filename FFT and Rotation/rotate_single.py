import math
import numpy as np
from lib import method as M
from numba import cuda,jit
import time
Pi=math.pi

import copy
import os


import argparse
parser=argparse.ArgumentParser(description='T')
parser.add_argument('--input_mrc',type=str,default='0mrc.mrc')
parser.add_argument('--angle',type=str,default='0,180,0')
parser.add_argument('--output_mrc',type=str,default='full.mrc')
parser.add_argument('--angpix',type=float,default=1)
parser.add_argument('--rot_seq',type=str,default='zxz')
parser.add_argument('--input_star',type=str,default='')

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
  #x,y,z=volume.shape
  #radius=int((x-1)/2)
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


@cuda.jit
def set_volume(Bigbox,little_volume,cx,cy,cz,radius):
  i,j,k=cuda.grid(3)
  #if  Bigbox[cx-radius+i,cy-radius+j,cz-radius+k]
  Bigbox[cx-radius+i,cy-radius+j,cz-radius+k]+=little_volume[i,j,k]  


def Remove(input_str,char):
  while(1):
    try:
      input_str.remove(char)
    except:
      return input_str

def get_col(attr,star):
  cmd="echo ` awk '{if ($1==\""+attr+"\") print $2}' "+star+" |sed 's/#//' ` "
  tmp_value=int(   os.popen(cmd).read().split('\n')[0]  )
  return tmp_value


#input format
#dx,dy,dz,phi,theta,psi,x,y,z
#


'''
ori_volume=M.read_float_mrc(args.input_mrc)
radius=int(ori_volume.shape[0]/2)


blockdim=ori_volume.shape
griddim=1
'''
Rot=get_col("_rlnAngleRot",args.input_star)

Tilt=get_col("_rlnAngleTilt",args.input_star)

Psi=get_col("_rlnAnglePsi",args.input_star)



#output_volume=np.zeros(shape=ori_volume.shape,dtype=ori_volume.dtype)

euler=[]
input_list=args.angle.split(',')
for x in input_list:
  euler.append(float(x)/180*Pi)




R=euler2matrix(euler,args.rot_seq)
#print(R)
#some scripts grep results:
# r =12
# t =13
# p =14
fp=open('J176.star')

context=fp.read().split('\n')
#print (len(context))

out=open('new.star','w+')


for eachline in context:
  line_split=eachline.split(' ')
  #print(line_split)
 # try:
 #   line_split.remove('')
 # except:
 #   line_split=line_split
  
 # while len(line_split)<10:
  
  #  continue
  try :
    rot_value=float(line_split[Rot-1])
    tilt_value=float(line_split[Tilt-1])
    psi_value=float(line_split[Psi-1])
    old_angle=np.matrix([float(rot_value), float(tilt_value) ,  float(psi_value) ])
    #print(old_angle)
    new_angle=list(np.array(old_angle*R)[0])
    #print(new_angle)
    new_rot=new_angle[0]
    new_tilt=new_angle[1]
    new_psi=new_angle[2]
    for x in range(len(line_split)):
      if x==Rot-1:
        out.write(str(format(new_rot,'.6f')))
      elif x==Tilt-1:
        out.write(str(format(new_tilt,'.6f')))
      elif x==Psi-1:
        out.write(str(format(new_psi,'.6f')))
      else:
        out.write(line_split[x])
      out.write(' ')
    out.write('\n')
    out.flush() 
    #cmd= "     awk      ' {$" +str(Rot)+"="+str(new_rot)+";$"+str(Tilt)+"="+str(new_tilt)+";$"+str(Psi)+"="+str(new_psi)+"; print }'    J176.star  "
    #print('x')
    #print(cmd)
  #  print(list(np.array(new_angle)[0]))
  except:
    for x in line_split: 
      out.write(x)
      out.write(' ')
    out.write('\n')
    out.flush()
    continue

#vector3=np.zeros([1,3],dtype=np.float32)
#o_vec=np.zeros([1,3],dtype=np.float32)

#calc[blockdim,griddim](ori_volume,R,output_volume,vector3,o_vec)


#M.write_pix_file(output_volume,args.output_mrc,args.angpix)


