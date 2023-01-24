#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.models import Model
from keras.models import load_model
from keras.layers import Input,GlobalAveragePooling2D,Layer,InputSpec
from keras.layers.core import Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.utils import np_utils
import keras.backend as K
import keras.layers as kl
import tensorflow as tf
from tensorflow.python.client import device_lib
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from time import time
import pickle
from AttentionModule import CrossAttention, SoftAttention, ResidualCombine2D
import os, sys, shutil
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2
from tqdm import tqdm_notebook
from sklearn.model_selection import KFold
from scipy import misc
from itertools import combinations
pd.options.display.max_colwidth = 100
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics



from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())



import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
n_gpu=0
n_cpu=0
tf_config= tf.compat.v1.ConfigProto(device_count = {'GPU': n_gpu , 'CPU': n_cpu})
tf_config.gpu_options.allow_growth=True
s=tf.Session(config=tf_config)
K.set_session(s)



ds_folder = '/home/nu/Work/Heba/Image_Forensics_Project/dataset/all'
ds = np.array(os.listdir(ds_folder))
np.random.shuffle(ds)
ds[:10]




def VGGNet():
    image_input = Input(shape=(64,64,1),name='image_input')
#     x = CoordinateChannel2D()(image_input)
    x = kl.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='block1_conv1')(image_input)
    x = kl.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', name='block1_conv2')(x)
    x = kl.BatchNormalization()(x)
    x_red = kl.MaxPooling2D(pool_size=(2,2), padding='same', name='block1_reduction_conv')(x)    
    xsa1,samap1 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(x.shape[-1]),name='soft_attention_1')(x)
    xsa1 = kl.MaxPooling2D(pool_size=(2,2),padding='same')(xsa1)
    x = kl.Concatenate()([x_red,xsa1])
#     x,sa1g1,sa1g2=ResidualCombine2D(ch_in=int(sa1.shape[-1]),ch_out=64)([x,sa1])
#     x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(0.5)(x)

    x = kl.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='block2_conv1')(x)
    x = kl.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', name='block2_conv2')(x)
    x = kl.BatchNormalization()(x)
    x_red = kl.MaxPooling2D(pool_size=(2,2), padding='same', name='block2_reduction_conv')(x)
    xsa2,samap2 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(x.shape[-1]),name='soft_attention_2')(x)
    xsa2 = kl.MaxPooling2D(pool_size=(2,2),padding='same')(xsa2)
    x = kl.Concatenate()([x_red,xsa2])
#     x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(0.5)(x)


    return Model(image_input,x,name='imgModel')



feat_ext_A = VGGNet()
feat_ext_A.summary()



SVG(model_to_dot(feat_ext_A,show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# ## extract features from the stem model



l_inp = kl.Input(shape=(64,64,1),name='l')
r_inp = kl.Input(shape=(64,64,1),name='r')
left_feats = feat_ext_A(l_inp)
right_feats = feat_ext_A(r_inp)


# ## apply cross attention between the extracted features



att_l_1,l_map_1 = CrossAttention(ch=int(left_feats.shape[-1]),name='ca_l2r_1')([left_feats,right_feats]) #[k,q]
att_l_1 = kl.BatchNormalization()(att_l_1)
att_l_2,l_map_2 = CrossAttention(ch=int(left_feats.shape[-1]),name='ca_l2r_2')([left_feats,right_feats]) #[k,q]
att_l_2 = kl.BatchNormalization()(att_l_2)
att_l_conc = kl.Concatenate(name='ca_l2r_concat')([att_l_1,att_l_2])
att_l_conc = kl.Activation('relu')(att_l_conc)
l2r_comb,l2r_g1,l2r_g2 = ResidualCombine2D(ch_in=int(att_l_conc.shape[-1]),ch_out=512,name='ca_l2r_comb')([left_feats,att_l_conc])
l2r_comb = kl.BatchNormalization()(l2r_comb)

att_r_1,r_map_1 = CrossAttention(ch=int(right_feats.shape[-1]),name='ca_r2l_1')([right_feats,left_feats])#[k,q]
att_r_1 = kl.BatchNormalization()(att_r_1)
att_r_2,r_map_2 = CrossAttention(ch=int(right_feats.shape[-1]),name='ca_r2l_2')([right_feats,left_feats])#[k,q]
att_r_2 = kl.BatchNormalization()(att_r_2)
att_r_conc = kl.Concatenate(name='ca_r2l_concat')([att_r_1,att_r_2])
att_r_conc = kl.Activation('relu')(att_r_conc)
r2l_comb,r2l_g1,r2l_g2 = ResidualCombine2D(ch_in=int(att_r_conc.shape[-1]),ch_out=512,name='ca_r2l_comb')([right_feats,att_r_conc])
r2l_comb = kl.BatchNormalization()(r2l_comb)


# ## combine the attended features



all_concat = kl.Concatenate()([l2r_comb,r2l_comb])
all_concat = kl.Activation('relu')(all_concat)


# ## futher joint feature extraction



x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv1')(all_concat)
x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv2')(x)
x = kl.Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same', name='block3_conv3')(x)
x = kl.BatchNormalization()(x)
x_red = kl.MaxPooling2D(pool_size=(2,2),padding='same', name='block3_reduction_conv')(x)
xsa3,samap3 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(x.shape[-1]),name='soft_attention_3')(x)
xsa3 = kl.MaxPooling2D(pool_size=(2,2),padding='same')(xsa3)
x = kl.Concatenate()([x_red,xsa3])
x = kl.Activation('relu')(x)
x = kl.Dropout(0.5)(x)

x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv1')(x)
x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv2')(x)
x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block4_conv3')(x)
x = kl.BatchNormalization()(x)
x_red = kl.MaxPooling2D(pool_size=(2,2),padding='same', name='block4_reduction_conv')(x)
xsa4,samap4 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(x.shape[-1]),name='soft_attention_4')(x)
xsa4 = kl.MaxPooling2D(pool_size=(2,2),padding='same')(xsa4)
x = kl.Concatenate()([x_red,xsa4])
x = kl.Activation('relu')(x)
x = kl.Dropout(0.5)(x)

x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv1')(x)
x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv2')(x)
x = kl.Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same', name='block5_conv3')(x)
x = kl.BatchNormalization()(x)
x_red = kl.MaxPooling2D(pool_size=(2,2), padding='same', name='block5_reduction_conv')(x)
xsa5,samap5 = SoftAttention(aggregate=True,m=16,concat_with_x=False,ch=int(x.shape[-1]),name='soft_attention_5')(x)
xsa5 = kl.MaxPooling2D(pool_size=(2,2),padding='same')(xsa5)
x = kl.Concatenate()([x_red,xsa5])
x = kl.Activation('relu')(x)
x = kl.Dropout(0.5)(x)

g = kl.Flatten()(x)
g = kl.Dropout(0.5)(g)
g = kl.Dense(2,activation='softmax',name='y')(g)
model = Model(inputs = [l_inp,r_inp],outputs = g)
model.summary()


# ## view the model in SVG mode



SVG(model_to_dot(model,show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))


# ## load data for a given fold



nb_fold = 1
df_tr = pd.read_csv('/home/nu/Work/Heba/Image_Forensics_Project/dataset/kfold_csvs/fold_1/train.csv'.format(nb_fold))
df_v = pd.read_csv('/home/nu/Work/Heba/Image_Forensics_Project/dataset/kfold_csvs/fold_1/val.csv'.format(nb_fold))
display(df_tr.shape,df_v.shape)
df_tr = df_tr.sort_values(by=['left','right','label']).reset_index()
df_v = df_v.sort_values(by=['left','right','label']).reset_index()


# ## visualize the ground truth data



display(df_tr.head())
df_tr.shape



display(df_v.head()),df_v.shape


# ## function for datagenerator



def datagen(ds,batch_size=128,seq=False,mode=''):
    counter=0
    l,r,y,names = [],[],[],[]
    idx = 0
    while True:
        i=np.random.randint(0,ds.shape[0],1)[0]
        if seq:
            i = idx
            idx+=1
        d = ds.iloc[i]        
        l.append((255.0 - np.expand_dims(cv2.imread(os.path.join(ds_folder,d.left),0),axis=-1))/255.0)
        r.append((255.0 - np.expand_dims(cv2.imread(os.path.join(ds_folder,d.right),0),axis=-1))/255.0)
#         im = np.concatenate((l,r),axis=-1)  
        names.append([d.left,d.right])
        y.append(d.label)
        counter+=1
        
        r.append((255.0 - np.expand_dims(cv2.imread(os.path.join(ds_folder,d.left),0),axis=-1))/255.0)
        l.append((255.0 - np.expand_dims(cv2.imread(os.path.join(ds_folder,d.right),0),axis=-1))/255.0)
#         im = np.concatenate((l,r),axis=-1)  
        names.append([d.right,d.left])
        y.append(d.label)
        
        counter+=1
#         print('---',mode,'---',names)
        if ds.shape[0]==idx:
            idx = 0
        if counter==batch_size:
            inputs={
                'l':np.array(l)
                ,'r':np.array(r)
            }
            outputs={
#                 'y':np.array(y)
                'y':np_utils.to_categorical(y,num_classes=2)
                ,'names':np.array(names)
            }
            yield inputs,outputs
            counter=0
            l,r,y,names = [],[],[],[]


# ## function for categorical focal loss


def categorical_focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        cross_entropy = -y_true*K.log(y_pred)
#         print(K.get_value(y_true))
        alpha_factor = K.ones_like(y_true)*alpha
        a_t = K.tf.where(K.equal(K.argmax(y_true),1),alpha_factor, 1.0-alpha_factor)
#         print(a)
        weight = a_t * y_true * K.pow((1-y_pred), gamma)
        loss = weight * cross_entropy
        loss = K.sum(loss, axis=1)
        return loss    
    return focal_loss


tr_batch_size=192
val_batch_size=512
tr_gen = datagen(df_tr,batch_size=tr_batch_size,seq=True,mode='train')
v_gen = datagen(df_v,batch_size=val_batch_size,seq=True,mode='val')




inputs,outputs=next(tr_gen)
l,r,y,n = inputs['l'],inputs['r'],outputs['y'],outputs['names']




l.shape,r.shape,y.shape,n.shape



plt.imshow(l[0][:,:,0])
plt.show()
plt.imshow(r[0][:,:,0])
plt.show()
plt.imshow(l[1][:,:,0])
plt.show()
plt.imshow(r[1][:,:,0])
plt.show()



p_l = feat_ext_A.predict(l)
p_l.shape



y_true = np.array([0.0,1.0])
y_pred = np.array([0.8,0.2])
cce = -y_true*np.log(y_pred)
cce



alpha = 0.95 if np.argmax(y_true)==1 else 1 - 0.95
print(alpha)
gamma = 2.0
w = alpha * y_true * np.power((1.0-y_pred), gamma)

w*cce


plt.imshow(p_l[0][:,:,92])


# # Experiments
# - images are input as concatenated
#     - equal class weights
#     - unbalanced class weights
#     - focal loss
# - images are input as siamese inputs
#     - equal class weights
#     - unbalanced class weights
#     - focal loss

# ## This is for unbalanced class



from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



model.compile(loss=categorical_focal_loss(),optimizer=Adam(lr=0.0001,decay=1e-06),metrics=['accuracy']


class MultiGPUCheckpoint(ModelCheckpoint):    
    def set_model(self, model):
        if isinstance(model.layers[-2], Model):
            self.model = model.layers[-2]
        else:
            self.model = model




if not os.path.exists('/home/nu/Work/Heba/Image_Forensics_Project/AttentionHandwritingVerification/checkpoints/ICFHR2020/'):
    os.mkdir('/home/nu/Work/Heba/Image_Forensics_Project/AttentionHandwritingVerification/checkpoints/checkpoints/ICFHR2020/')
mc=MultiGPUCheckpoint(filepath='/home/nu/Work/Heba/Image_Forensics_Project/AttentionHandwritingVerification/checkpoints/checkpoints/ICFHR2020/cross_attention_residual_vgg_fold_{1}.h5'.format(nb_fold),monitor='val_loss',period=1,save_best_only=True,save_weights_only=True,mode='auto',verbose=1)
es=EarlyStopping(patience=20,monitor='val_loss',min_delta=0.0001,mode='auto')
tb=TensorBoard(log_dir="/home/nu/Work/Heba/Image_Forensics_Project/AttentionHandwritingVerification/checkpoints/runs/cross_attention_residual_vgg_fold_{1}".format(nb_fold,time()))


# ## load weights if any


model.load_weights('/home/nu/Work/Heba/Image_Forensics_Project/AttentionHandwritingVerification/checkpoints/ICFHR2020/cross_attention_residual_vgg_fold_1.h5',by_name=True,skip_mismatch=True)


# ## start the training


#Epochs=100
# Epochs=10             
Epochs=5
h=model.fit_generator(tr_gen,initial_epoch=5,callbacks=[mc,es,tb],epochs=Epochs,steps_per_epoch=2*((df_tr.shape[0]//tr_batch_size)+1)
                      ,validation_data=v_gen, validation_steps=2*60, verbose=1)
                               )


# # Evaluation



model.load_weights('/home/nu/Work/Heba/Image_Forensics_Project/AttentionHandwritingVerification/checkpoints/ICFHR2020/cross_attention_residual_vgg_fold_1.h5')



v_gen = datagen(df_v,batch_size=256,seq=True)
inputs,outputs=next(v_gen)
test_x,test_y = inputs,outputs['y']


# ## Make post processing models to visualize attention maps



soft_attention_1 = Model(feat_ext_A.inputs,feat_ext_A.get_layer('soft_attention_1').output)
feats_sa1,maps_sa1 = soft_attention_1.predict(test_x['l'])
soft_attention_2 = Model(feat_ext_A.inputs,feat_ext_A.get_layer('soft_attention_2').output)
feats_sa2,maps_sa2 = soft_attention_2.predict(test_x['l'])
soft_attention_3 = Model(model.inputs,model.get_layer('soft_attention_3').output)
feats_sa3,maps_sa3 = soft_attention_3.predict(test_x)
soft_attention_4 = Model(model.inputs,model.get_layer('soft_attention_4').output)
feats_sa4,maps_sa4 = soft_attention_4.predict(test_x) 
soft_attention_5 = Model(model.inputs,model.get_layer('soft_attention_5').output)
feats_sa5,maps_sa5 = soft_attention_5.predict(test_x)




ca_l2r_comb_model = Model(model.inputs,model.get_layer('ca_l2r_comb').output)
f_ca_l2r_comb,l2r_g1,l2r_g2 = ca_l2r_comb_model.predict(test_x)

ca_r2l_comb_model = Model(model.inputs,model.get_layer('ca_r2l_comb').output)
f_ca_r2l_comb,r2l_g1,r2l_g2 = ca_r2l_comb_model.predict(test_x)

ca_l2r_1_model = Model(model.inputs,model.get_layer('ca_l2r_1').output)
f_ca_l2r_1,maps_l2r_1 = ca_l2r_1_model.predict(test_x)

ca_l2r_2_model = Model(model.inputs,model.get_layer('ca_l2r_2').output)
f_ca_l2r_2,maps_l2r_2 = ca_l2r_2_model.predict(test_x)

ca_r2l_1_model = Model(model.inputs,model.get_layer('ca_r2l_1').output)
f_ca_r2l_1,maps_r2l_1 = ca_r2l_1_model.predict(test_x)

ca_r2l_2_model = Model(model.inputs,model.get_layer('ca_r2l_2').output)
f_ca_r2l_2,maps_r2l_2 = ca_r2l_2_model.predict(test_x)


# ## check how much importance is being given to attention by the model



l2r_g1[image_idx],l2r_g2[image_idx],r2l_g1[image_idx],r2l_g2[image_idx]


# ## combine the left maps and right maps respectively


mapl  = maps_l2r_1+maps_l2r_2
mapl.shape
mapr  = maps_r2l_1+maps_r2l_2
mapr.shape




feats_sa3.shape,maps_sa3.shape, f_ca_l2r_1.shape,maps_l2r_1.shape

image_index = 23
print(outputs['names'][image_index])

preds = model.predict(test_x)

preds[image_index]


# ## Get important pixels to query



resized_left_img = cv2.resize(test_x['l'][image_index],(16,16),cv2.INTER_CUBIC)
resized_right_img = cv2.resize(test_x['r'][image_index],(16,16),cv2.INTER_CUBIC)





plt.imshow(resized_left_img)
plt.show()
plt.imshow(resized_right_img)
plt.show()





def getRowCol(point):
    row = point//16
    col = point%16
    return row,col


# # Visualize Cross Attention



'''
To visualize a right image query pixel (query image) 'q_pix' 
that matches with k pixels of left image (key image)
- chose map_r2l_<x>
- select a row 'r' in it.
- reshap to 16x16 and resize to 64x64.
- apply as mask on left image.
'''

shape = (16,16)
right_useful_pixels = []
for i in range(len(resized_right_img)):
    for j in range(len(resized_right_img)):
        if resized_right_img[i][j] > 0:
            right_useful_pixels.append([i,j])
            
left_useful_pixels = []
for i in range(len(resized_left_img)):
    for j in range(len(resized_left_img)):
        if resized_left_img[i][j] > 0:
            left_useful_pixels.append([i,j])
            
print('# of query pixels (useful pixels from right image):',len(right_useful_pixels))
f, axarr = plt.subplots(nrows=len(right_useful_pixels),ncols=2,figsize=(6,3*len(right_useful_pixels)))

counter=0
left_points = 0
for i,j in right_useful_pixels:# no transparent/black pixels
#     print(i,j)
    pixel_num = (i*16)+j
#     print('pixel_num:',pixel_num)
    argmax=np.argmax(mapl[image_index][pixel_num])
    prob=mapl[image_index][pixel_num][argmax]
#     print('argmax:',argmax)
    row,col = getRowCol(argmax)
#     print([row,col])
    if [row,col] in left_useful_pixels:
        left_points += 1
        left_useful_pixels.remove([row,col])
    
    key_point = np.zeros((16,16))
    key_point[i][j] = 1.0
    resized_key_point = cv2.resize(key_point,(64,64),cv2.INTER_CUBIC)
    axarr[counter,0].imshow(resized_key_point,cmap='gray')
    axarr[counter,0].imshow(np.squeeze(test_x['r'][image_index],-1),alpha=0.3,cmap='gray')
    
    axarr[counter,1].imshow(cv2.resize(mapl[image_index][pixel_num].reshape(shape),(64,64),interpolation=cv2.INTER_CUBIC),cmap='gray')
    axarr[counter,1].imshow(np.squeeze(test_x['l'][image_index],-1),alpha=0.3,cmap='gray')
    axarr[counter,0].set_title('right pxl:{0},{1}'.format(i,j))
    axarr[counter,1].set_title('left pxl:{0},{1}, prob:{2:.4f}'.format(row,col,prob))
    axarr[counter,0].axis('off')
    axarr[counter,1].axis('off')
    counter+=1
    
print('matched left_points:',left_points)


# ## See all CA maps



plt.imshow(maps_l2r_1[image_index],cmap='jet')
plt.show()
plt.imshow(maps_l2r_2[image_index],cmap='jet')
plt.show()
plt.imshow(maps_r2l_1[image_index],cmap='jet')
plt.show()
plt.imshow(maps_r2l_2[image_index],cmap='jet')
plt.show()





print('-----**********visualize SA1 maps************-----')
plt.axis('off')
plt.imshow(cv2.resize(maps_sa1[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_CUBIC),cmap='jet')
plt.imshow(np.squeeze(test_x['l'][image_index],-1),cmap='gray',alpha=0.3)
plt.show()
print("================================")
plt.axis('off')
plt.imshow(cv2.resize(maps_sa1[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_CUBIC),cmap='jet')
plt.imshow(np.squeeze(test_x['r'][image_index],-1),cmap='gray',alpha=0.3)
plt.show()

print('-----**********visualize SA2 maps************-----')
plt.axis('off')
plt.imshow(np.squeeze(test_x['l'][image_index],-1))
plt.imshow(cv2.resize(maps_sa2[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_CUBIC),cmap='jet',alpha=0.5)
plt.show()
print("================================")
plt.axis('off')
plt.imshow(np.squeeze(test_x['r'][image_index],-1))
plt.imshow(cv2.resize(maps_sa2[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_CUBIC),cmap='jet',alpha=0.5)
plt.show()

print('-----**********visualize SA3 maps************-----')
plt.axis('off')

plt.imshow(cv2.resize(maps_sa3[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_LINEAR),cmap='jet')
plt.imshow(np.squeeze(test_x['l'][image_index],-1),cmap='gray',alpha=0.3)
plt.show()
print("================================")
plt.axis('off')

plt.imshow(cv2.resize(maps_sa3[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_LINEAR),cmap='jet')
plt.imshow(np.squeeze(test_x['r'][image_index],-1),cmap='gray',alpha=0.3)
plt.show()



print('-----**********visualize SA4 maps************-----')
plt.axis('off')
plt.imshow(np.squeeze(test_x['l'][image_index],-1))
plt.imshow(cv2.resize(maps_sa4[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_LINEAR),cmap='jet',alpha=0.5)
plt.show()
print("================================")
plt.axis('off')
plt.imshow(np.squeeze(test_x['r'][image_index],-1))
plt.imshow(cv2.resize(maps_sa4[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_LINEAR),cmap='jet',alpha=0.5)
plt.show()

print('-----**********visualize SA5 maps************-----')
plt.axis('off')
plt.imshow(np.squeeze(test_x['l'][image_index],-1))
plt.imshow(cv2.resize(maps_sa5[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_CUBIC),cmap='jet',alpha=0.5)
plt.show()
print("================================")
plt.axis('off')
plt.imshow(np.squeeze(test_x['r'][image_index],-1))
plt.imshow(cv2.resize(maps_sa5[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_LINEAR),cmap='jet',alpha=0.5)
plt.show()





maps_sa5[image_index].sum(axis=0)

# sn.heatmap(maps2_r[image_index][j+i*26].reshape((26,26)),cmap='jet')
sn.heatmap(cv2.resize(maps_sa3[image_index].sum(axis=0),(64,64),interpolation=cv2.INTER_CUBIC),cmap='jet')
    


# # Evaluate the network
# - Evaluating network VGG Cross and Soft Attention


classnames=['Inter','Intra']




v_gen = datagen(df_v,batch_size=512,seq=True)
arr_test_y = []
arr_preds = []
for i in tqdm_notebook(range(2*(1+int(df_v.shape[0]/512)))):
    inputs,outputs=next(v_gen)
    test_x,test_y = inputs,outputs['y']
    predictions = parallel_model.predict(test_x)
    arr_test_y+=list(test_y)
    arr_preds+=list(predictions)
arr_test_y = np.array(arr_test_y)
arr_preds = np.array(arr_preds)
print(arr_test_y.shape)
print(arr_preds.shape)



arr_preds_logits = arr_preds.argmax(-1)
print(arr_preds[image_index],arr_preds_logits[image_index],arr_test_y[image_index])
# arr_preds_logits = np.array([1 if a>0.5 else 0 for a in arr_preds])
# arr_preds_logits,arr_test_y.argmax(-1)



df = pd.DataFrame()
f = classification_report(arr_test_y.argmax(-1),arr_preds_logits,target_names=classnames,output_dict=True)
df = df.from_dict(f)
display(df.T)




cm=metrics.confusion_matrix(arr_test_y.argmax(-1),arr_preds_logits)
cm


def get_FPR_intra(cm):
    FP = cm[0][1]
    TN = cm[0][0]
    return FP/(FP+TN)
def get_FNR_intra(cm):
    FN = cm[1][0]
    TP = cm[1][1]
    return FN/(FN+TP)



get_FPR_intra(cm)*100,get_FNR_intra(cm)*100



def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(3, 3))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()





plot_confusion_matrix(cm=cm,cmap='YlOrBr',target_names=classnames, normalize=False)

