
import numpy as np
import cv2
#from google.colab.patches import cv2_imshow
from  matplotlib import pyplot as plt
import random
from PIL import Image as im



def luminiance_remap(l_s, l_t):

    '''luminance remapping: faire une correspondance entre les valeurs de luminance
       Input : l channel de l'image de reference, l channel du target
       calculer la moyenne et l'écart type
       changer L channel en utilisant la mean et std
       Output : remapped l_s channel
    '''
     
    mu_s = np.mean(l_s)
    mu_t = np.mean(l_t)
    sigma_s = np.std(l_s)
    sigma_t = np.std(l_t)
    remapped = (l_s - mu_s)* (sigma_t / sigma_s) + mu_t
    
    return remapped



def random_sampling(source_image,number_of_samples):

    '''Collecter des échantillons aléatoires
       Input: image source, nombre d'échantillon
       Output: cellules de pixels de taille nxn
    '''
    
    #taille de cellule
    n = int(np.sqrt(number_of_samples))

    #taille de l'image source
    y, x = source_image[:,:,0].shape

    #step sur les deux axes x et y
    step_y = y // n
    step_x = x // n
    
    sampled = np.zeros([n,n,3])
    
    for i in range(0,n):
        for j in range(0,n):
            
            actual_i = random.randrange(i*step_y, (i+1)*step_y-1, 1)
            actual_j = random.randrange(j*step_x, (j+1)*step_x-1, 1)
            
            sampled[i,j,:] = source_image[actual_i,actual_j,:]
            
    return sampled



def sd_neighbourhood(l_channel, neighbourhood_size):

    '''calculer l'écart type pour chaque pixel dans un voisinage de 5x5
       Input : L channel de l'image source et target, taille de voisinage (5)
       Output : sds - l'écart type de L channel
    '''
    
    amt_to_pad = (neighbourhood_size - 1) // 2

    y, x = l_channel.shape
    sds = np.zeros(l_channel.shape)

    #padding the image for boundary pixels
    padded = np.pad(l_channel, (amt_to_pad, amt_to_pad))
    
    for i in range(amt_to_pad+1,y+2):
        for j in range(amt_to_pad+1,x+2):
            region = padded[i-2:i+2, j-2:j+2]
            sd = np.std(region[:])
            sds[i-2, j-2] = sd
            
    return sds



def color_transfer(source,target,source_std,target_std):

    '''transferer les couleur de l'image en gris
       Input : source image, target image
               écart type des pixels d'échantillons
               écart type des pixels de target
       Output : image colorée
    '''

    #extraire les valeurs L, a et b de l'image source et traget dans l'espace LAB
    (l_t,a_t,b_t) = cv2.split(target)
    (l_s,a_s,b_s) = cv2.split(source)
    
    y, x = l_t.shape
    
    for i in range(y):
        for j in range(x):
            
            weighted_sum = 0.5 * np.square(l_s - l_t[i,j]) + 0.5 * np.square(source_std - target_std[i,j])

            #finding the pixel with minimum weighted sum and transferring a b pixel values
            index = np.argwhere(weighted_sum == np.min(weighted_sum))
        
            a_t[i,j] = a_s[index[0,0],index[0,1]]
            b_t[i,j] = b_s[index[0,0],index[0,1]]
            
    transformed = cv2.merge([l_t,a_t,b_t])
    transformed = cv2.cvtColor(transformed.astype("uint8"),cv2.COLOR_LAB2BGR)
    
    return transformed

image = cv2.imread('C:/Users/Tema/Downloads/imagecolor.jpg') #image de référence
image2 = cv2.imread('C:/Users/Tema/Downloads/gray.png') #image en niveaux de gris

######### 1ere étape
#converitr l'image de base de référence en LAB
source = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

#converitr l'image en niveaux de gris en LAB
target = cv2.cvtColor(image2, cv2.COLOR_BGR2LAB)



##### 2eme étape
#luminance remapping
source[:,:,0] = luminiance_remap(source[:,:,0],target[:,:,0])
# interval of values [0:255]
source[:,:,0] = np.clip(source[:,:,0], 0, 255)

######### 3eme étape
#choisir un échantillon de pixels aléatoire
# nombre d'échantillons = 200
sampled_source = random_sampling(source,200)



########## 4eme étape
#calculer l'écart type pour chaque pixel dans un voisinage de 5x5
source_std = sd_neighbourhood(sampled_source[:,:,0],5)
target_std = sd_neighbourhood(target[:,:,0],5)

######## 5th étape
#transferer les couleur de l'image en gris
final_image = color_transfer(sampled_source,target,source_std,target_std)

cv2_imshow(final_image)



