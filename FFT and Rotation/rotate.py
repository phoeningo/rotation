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
parser.add_argument('--input_table',type=str,default='volume8.log')
parser.add_argument('--output_mrc',type=str,default='full.mrc')
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


#input format
#dx,dy,dz,phi,theta,psi,x,y,z
#
def read_dynamo_transform(input_file):
  input_fp=open(input_file)
  input_read=input_fp.read()
  input_split=input_read.split('\n')
  L=len(input_split)
  tmp_euler=[]
  tmp_coords=[]
  for linei in range(L-1):
    lines=input_split[linei]
    line_split=lines.split(' ')
    if (len(line_split)>1):
      line_remove=Remove(line_split,'')
      tmp_euler.append([ float(line_remove[3])/180*Pi,float(line_remove[4])/180*Pi,  float(line_remove[5])/180*Pi  ])
      tmp_coords.append(  [ float(line_remove[6])+float(line_remove[0]),float(line_remove[7])+float(line_remove[1]), float(line_remove[8])+float(line_remove[2])]   )

  return tmp_euler,np.array(tmp_coords)


t1=time.time()

euler,coords=read_dynamo_transform(args.input_table)




ori_volume=M.read_float_mrc(args.input_mrc)
full_size=ori_volume.shape[0]
radius=int(full_size/2)
input_angle=Pi/2,3/4*Pi,0


blockdim=ori_volume.shape
griddim=1

X=coords[:,0]
Y=coords[:,1]
Z=coords[:,2]

start_x=min(X)
end_x=max(X)
x_range=end_x-start_x

start_y=min(Y)
end_y=max(Y)
y_range=end_y-start_y

start_z=min(Z)
end_z=max(Z)
z_range=end_z-start_z

#print(x_range,y_range,z_range)
out_box_size=int(max(x_range,y_range,z_range)+radius*2)
out_radius=int(out_box_size/2)
output_full_volume=np.zeros([out_box_size,out_box_size,out_box_size],dtype=ori_volume.dtype)

#so ,transform vector is -1*(start_x,start_y,start_z),

#print (X)
print('Init...')

for vi in range(8):
  
  output_volume=np.zeros([2*radius,2*radius,2*radius],dtype=ori_volume.dtype)
  R=euler2matrix(euler[vi],args.rot_seq)
  #print(R)
  cdz,cdy,cdx=out_radius-int(x_range/2)+int(coords[vi,0]-start_x),out_radius-int(y_range/2)+int(coords[vi,1]-start_y),out_radius-int(z_range/2)+int(coords[vi,2]-start_z)

  t_bd=radius*2,radius*2,radius*2
  t_gd=1

  vector3=np.zeros([1,3],dtype=np.float32)
  o_vec=np.zeros([1,3],dtype=np.float32)

  calc[blockdim,griddim](ori_volume,R,output_volume,vector3,o_vec)
  set_volume[t_bd,t_gd](output_full_volume,output_volume,cdx,cdy,cdz,radius)
  print('Writing ...'+str(vi+1)+'/ 8 subvolume')
  #M.write_pix_file(output_volume,'sub'+str(vi)+'.mrc',1)


M.write_pix_file(output_full_volume,args.output_mrc,args.angpix)
print(time.time()-t1)


