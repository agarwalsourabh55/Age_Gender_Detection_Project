# Age_Gender_Detection<br/>

This project is all about detecing age and gender of the human face based on UTKFace collection Where approx on 23000 images computaion is performed .<br/>
So Project File and dataset is attached in this repository 

Contributors : <br/>
Sourabh Agarwal<br/>
Shailesh Goyal

Dataset-https://susanqq.github.io/UTKFace/<br/>
UTKFace dataset is a large-scale face dataset with long age span, which ranges from 0 to 116 years old. The images cover large variation in pose, facial<br/> expression, illumination, occlusion, resolution and other such.<br/>

Size: The dataset consists of over 20K images with annotations of age, gender and ethnicity.<br/>

Projects: The dataset can be used on a variety of task such as facial detection, age estimation, age progression, age regression, landmark localisation, etc. <br/>



#################################
import cv2   <br/>
images = []<br/>
ages = []<br/>
genders = []<br/>
for file in files:<br/>
    image = cv2.imread(path+file,0)<br/>
    image = cv2.resize(image,dsize=(64,64))<br/>
    image = image.reshape((image.shape[0],image.shape[1],1))<br/>
    images.append(image)<br/>
    split_var = file.split('_')<br/>
    ages.append(split_var[0])<br/>
    genders.append(int(split_var[1]))<br/>
####################################
In this part of code I just change the size of the image so that we all can have same size of image this will help in compuation    <br/>

####################################
def display(img):<br/>
    plt.imshow(img[:,:,0])<br/>
    plt.set_cmap('gray')<br/>
    plt.show()<br/>
idx = 500<br/>
sample = images[idx]<br/>
print("Gender:",genders[idx],"Age:",ages[idx])<br/>
display(sample)<br/>
####################################
Performing some data analysis in this part of code based on the age factor <br/>
Now later on we will predict on some image using CNN<br/>

Model: "functional_1"<br/>
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 64, 64, 1)]  0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 62, 62, 32)   320         input_1[0][0]                    
__________________________________________________________________________________________________
conv2d_1 (Conv2D)               (None, 60, 60, 64)   18496       conv2d[0][0]                     
__________________________________________________________________________________________________
max_pooling2d (MaxPooling2D)    (None, 30, 30, 64)   0           conv2d_1[0][0]                   
__________________________________________________________________________________________________
conv2d_2 (Conv2D)               (None, 28, 28, 128)  73856       max_pooling2d[0][0]              
__________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)  (None, 14, 14, 128)  0           conv2d_2[0][0]                   
__________________________________________________________________________________________________
dropout (Dropout)               (None, 14, 14, 128)  0           max_pooling2d_1[0][0]            
__________________________________________________________________________________________________
flatten (Flatten)               (None, 25088)        0           dropout[0][0]                    
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 128)          3211392     flatten[0][0]                    
__________________________________________________________________________________________________
dropout_2 (Dropout)             multiple             0           dense_4[0][0]                    
                                                                 dense_5[0][0]                    
                                                                 dense_6[0][0]                    
                                                                 dense_7[0][0]                    
                                                                 dense_8[0][0]                    
__________________________________________________________________________________________________
dense_5 (Dense)                 (None, 64)           8256        dropout_2[0][0]                  
__________________________________________________________________________________________________
dense (Dense)                   (None, 128)          3211392     flatten[0][0]                    
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 32)           2080        dropout_2[1][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             multiple             0           dense[0][0]                      
                                                                 dense_1[0][0]                    
                                                                 dense_2[0][0]                    
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           8256        dropout_1[0][0]                  
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 16)           528         dropout_2[2][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 32)           2080        dropout_1[1][0]                  
__________________________________________________________________________________________________
dense_8 (Dense)                 (None, 8)            136         dropout_2[3][0]                  
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 1)            33          dropout_1[2][0]                  
__________________________________________________________________________________________________
dense_9 (Dense)                 (None, 1)            9           dropout_2[4][0]                  
==================================================================================================
Total params: 6,536,834<br/>
Trainable params: 6,536,834<br/>
Non-trainable params: 0<br/>
__________________________________________________________________________________________________


#so after calling fit model we can finaaly fit our CNN network with the dataset <br/>
and after this we can finally predict what our output is <br/>

![alt text](https://github.com/agarwalsourabh55/Age_Gender_Detection_Project/blob/master/image.png?raw=true)<br/>









