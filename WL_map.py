#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:15:23 2026

@author: adamgrosset
"""
#%%
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
import re
from copy import copy
from scipy.signal import find_peaks
from matplotlib.ticker import MaxNLocator

####################################################################################
#########################       Settings        ####################################
####################################################################################
save_im=False

################ Neon image (line 26) ####################
#Path
Im_folder = "Chips_data/Teem_3D/IR/Im_teem3D/neon/Im_neon"
Dark_folder = "Chips_data/Teem_3D/IR/Im_teem3D/neon/Dark_neon"
#Title
img_title='Img_Neon'
plt_title='Neon_Peaks'
#Line selection for calibration (line 187)
treshold_detection_line = 300
Delta = 4
#Draw the selected lines in red 
line_in_red=True

################ Fiber image (line 193) ################
#Path
a = fits.getdata("Chips_data/Teem_3D/IR/2026_04_09/capture/object/Fiber_4_TM.fits")
dark = fits.getdata("Chips_data/Teem_3D/IR/2026_04_09/capture/dark/Fiber_4_TM_dark.fits")

################ Reference line (line 233) ################
#Path
Fiber=fits.getdata("Chips_data/Teem_3D/IR/2026_04_09/capture/object/Fiber_alone_2.fits")
Fiber_dark=fits.getdata("Chips_data/Teem_3D/IR/2026_04_09/capture/dark/Fiber_alone_dark.fits")

################ Where to save the plots ################
save_plots=("Chips_data/Teem_3D/red_data/Fiber_4_white_src") 

################ Plots title ################
Fiber_im = "Fiber_4"
Fiber_calibrate = "Fiber_4_cal"
Flux_distribution = "Flux_distribution_Input_4"
Chip_transmission = "Transmission_input_4"

####################################################################################
############################       WL map     ######################################
####################################################################################

# 1. Data retrieval, this function retrieves all file that end with .fits 
# and associate them with their dark

def Im_Dark (Im_folder,Dark_folder) :

    fits_list_Im = [f for f in os.listdir(Im_folder) if f.endswith('.fits')]
    fits_list_Dark = [f for f in os.listdir(Dark_folder) if f.endswith('.fits')]

    temp_Im = {} 
    temp_Dark = {}
    
    #boucles pour trouver le numéro dans le nom des fichiers et faire la moyenne 
    for p in fits_list_Im :
        
        num = ''.join(re.findall(r'\d+', p))
        fits_data_Im = fits.getdata(os.path.join(Im_folder,p))
        temp_Im[num]= np.mean(fits_data_Im, axis=0)

    for d in fits_list_Dark :
        num = ''.join(re.findall(r'\d+', d))
        fits_data_Dark = fits.getdata(os.path.join(Dark_folder,d))
        temp_Dark[num]= np.mean(fits_data_Dark, axis=0)

    Mean_dict = {}

    common_keys = set(temp_Im.keys()) & set(temp_Dark.keys())
    
    for key in common_keys:
        Mean_dict[key] = temp_Im[key] - temp_Dark[key]
     
    #ranger les clefs dans le bon ordre 
    corrected_images = dict(sorted(Mean_dict.items(), key=lambda x: int(x[0])))
    
    #somme de toute les im
    all_images_list = list(corrected_images.values())
    all_Im = np.sum(all_images_list, axis=0)
    
    print(Mean_dict.keys())
    return all_Im

Sum_im=Im_Dark(Im_folder, Dark_folder)

# 3. Affichage
def img_show(im_to_show,y_label,x_label,plot_title,save,save_plots_path,intensity_min,intensity_max):
    plt.figure(figsize=(10, 8))
    plt.imshow(im_to_show, origin='lower', vmin=intensity_min, vmax=intensity_max)
    plt.ylabel(y_label,fontsize=16)
    plt.xlabel(x_label,fontsize=16)
    plt.colorbar()
    plt.title(plot_title,fontsize=16)
    if save==True : 
        plt.savefig(os.path.join(save_plots_path, f"{plot_title}.png"),
                    bbox_inches='tight', pad_inches=0.1)

cal_img=img_show(Sum_im,y_label='Px',x_label='Px',plot_title=img_title,
                 intensity_min=0,intensity_max=100,
                 save=False,save_plots_path=save_plots)

# 4. trouver les lignes pour faire l'axe des x
def find_line(intresting_img,tresh,delt,removed_lines,red_line,want_to_plot,x_label,y_label,
              plot_title,x_max,y_max,x_min=0,y_min=0):

    "look at every line and take those that have at least one pixel up to 300"    
    line = []               #liste à remplir avec le numéro de la ligne
    line_copie = []

    for i in range(intresting_img.shape[0]):       #boucle pour trouver les lignes         
        if np.any(intresting_img[i, :]>tresh):     #valeurs de pixels superieur à la valuer donnée
            line.append(i)                         #rempli la liste
            line_copie.append(i)
    if removed_lines==True:       
        line = [o for o in line if not (o==4 or o == 133 or o == 134 or o == 222 or o==362 or o == 418 or o==457 or o==542 or o==715 or o==800)]
        line_copie = [o for o in line_copie if not (o==4 or o == 133 or o == 134 or o == 222 or o==362 or o == 418 or o==457 or o==542 or o==715 or o==800)]
    
    "look at every value on the list line and test if they are closer by 5 pixels or less"    
    line_index = []
    line_temp = []              #liste temporaire 

    delta = delt                #peut etre changer en fonction de combien de pixel font les faisceaux
    line_mean = []              #liste pour récuperer les valeurs moyennes 
             #liste pour les nom de chaques courbes 

    for l in line :
        if l in line_copie :      
            for i in range (l-delta,l+delta):  
                if i in line and i in line_copie:      
                    #print(i)
                    line_temp.append(i)     #remplissage de la liste temporaire 
                    line_copie.remove(i)    #enleve les valeurs deja tester de la liste line_copie           


        if len(line_temp)!=0 :                                  #on regarde si line temp est !=0
            miniline = np.mean(intresting_img[line_temp,:],axis=0)
            line_mean.append(miniline)     
            line_index.append(copy(line_temp)) 
            
        line_temp = []                      #réinitialisation de la liste temporaire 

    line_array=np.array(line_mean)              #Tansformation en array
    
#Trace selected line in red 
    if red_line==True :
        for k in line_index:
            #print(k)
            for h in k:
                plt.plot([h]*1200,'r', linewidth = 0.5)       
    
        plt.show()
    if want_to_plot==True :
        plt.figure(figsize=(10, 5))
        plt.title('Neons_spectra',fontsize=16)
        plt.plot(np.mean(line_array,axis=0))
        plt.ylabel('Intensity[ADU]',fontsize=16) 
        plt.xlabel('Px',fontsize=16)
        plt.grid(True)
    plt.figure(figsize=(10, 5))
    for p in range(line_array.shape[0]):
        plt.plot(line_array[p], label=f"output {p+1}")
        plt.title(plot_title,fontsize=16)
        plt.legend()
        plt.axis([x_min,x_max,y_min,y_max])
        plt.grid(True)
        plt.ylabel(y_label,fontsize=16) 
        plt.xlabel(x_label,fontsize=16)

    return line_array

# 5. axe de calibration
axis_cal = find_line(Sum_im,tresh=treshold_detection_line,delt=Delta,x_max=1280,y_max=1000,
                     red_line=line_in_red,removed_lines=True,
                     x_label='Px',y_label='Intensity[ADU]',plot_title=plt_title,want_to_plot=True)

# 6. crée l'axe x en matchant les pics avec ceux du neon théorique 
#Il faut rentrer manuellement les valeurs de neon en les comprant avec la fonction neon théorie

def create_x_axis(cal_line,want_to_trace,prominence_find_peaks=100):

    #find_peaks
    line1D = np.sum(cal_line,axis=0)
    #cal_line.reshape(-1)
    peaks,_ = find_peaks(line1D,prominence=prominence_find_peaks)
    plt.figure(figsize=(10, 5))
    print('Peaks_for_maping',peaks)
    
    # tracer le signal
    plt.plot(line1D, label="Signal")
    
    # tracer les pics
    plt.scatter(peaks, line1D[peaks], color='red', label="Pics")
    plt.axis([0,1250,0,6000])
    plt.title("find_peaks")
    plt.legend()
    plt.grid(True)
    plt.show()

    pixels_ref = np.array([53,124,138,227,238,302,307,342,369,507,625,695,763,857])
    ne_line_ref = np.array([1056,1079,1084,1114,1117,1139,1140,1152,1160,1198,1245,1268,1291,1321])
    
    ajust = np.polyfit(pixels_ref,ne_line_ref,1)
 
    ply = np.poly1d(ajust)
    axis = np.arange(0,1280)
    full_axis = ply(axis)
    #print(full_axis)
    if want_to_trace==True:
        plt.figure(figsize=(10,5))
        plt.plot(full_axis,line1D)
    return full_axis

axis_x = create_x_axis(axis_cal,want_to_trace=False)

# 7. récupération des données qui nous interesse et moyenne de a et soustraction du dark
def Mean (Mean_im, Dark_Mean):
    Mean_im = np.mean(Mean_im,axis=0)       #moyenne
    Dark_Mean = np.mean(Dark_Mean,axis=0)
    Im=Mean_im-Dark_Mean                    #soustraction du dark   

    return(Im)

lum = Mean(a,dark)

#afficher l'img de la puce que l'on veut calibrer        
img = img_show(lum,y_label='Px',x_label='Px',plot_title=Fiber_im,
               intensity_min=0,intensity_max=800,
               save=save_im,save_plots_path=save_plots)

#trouver les lignes qui nous intersse sur l'image 
usefull_line = find_line(lum,tresh=800,delt=10,x_max=1280,y_max=3000,
                        red_line=True,removed_lines=False,
                        x_label='Px',y_label='Intensity[ADU]',plot_title='Fiber_4',want_to_plot=False)

 
def final_plot (line_to_plot,save,save_plots_path,plot_title) :
    plt.figure(figsize=(10,5))
    for j in range(line_to_plot.shape[0]):
        plt.plot(axis_x,line_to_plot[j],label=f"output {j+1}")
        plt.axis([1050,1450,0,3000])
        plt.title(plot_title,fontsize=16)
        plt.ylabel('Intensity[ADU]',fontsize=16)
        plt.xlabel('Wavelength[nm]',fontsize=16)
        plt.legend()
        plt.grid(True)
        if save==True : 
            plt.savefig(os.path.join(save_plots_path, f"{plot_title}.png"),
                    bbox_inches='tight', pad_inches=0.1)

final_plot(usefull_line,save=save_im,save_plots_path=save_plots,
           plot_title=Fiber_calibrate)

####################################################################################
############################    Transmission  ######################################
####################################################################################

##########Reference line

Fiber_alone=Mean(Fiber, Fiber_dark)

Im_fiber_alone= img_show(Fiber_alone,y_label='Px',x_label='Px',plot_title='Fiber_alone',
               intensity_min=0,intensity_max=1000,
               save=False,save_plots_path=save_plots)

Lines_fiber_alone = find_line(Fiber_alone,tresh=2000, delt=15,
                              removed_lines=False, red_line=False,
                              x_label='Px', y_label='Intensity[ADU]', 
                              plot_title='Fiber_alone', x_max=1280, y_max=2000,want_to_plot=False)

############Interesting line 

def Distribution(save,save_plots_path,plot_title):
    
    # Normalisation par colonne
    col_sum = np.sum(usefull_line, axis=0)
    
    ratio = np.divide(usefull_line,col_sum)
    
    Mean_ratio = np.mean(ratio[:, :1280], axis=1)

    plt.figure(figsize=(10,5))
    plt.plot(Mean_ratio)
    plt.show()
    
    print(Mean_ratio,'akhqgajsfiy')
    
    x = range(1, usefull_line.shape[0] + 1)
    
    colors = ['C0','C1','C2','C3','C4','C5','C6','C7'] 
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, Mean_ratio*100, color=colors[:usefull_line.shape[0]])
    
    for bar, value in zip(bars, Mean_ratio):
        plt.text(bar.get_x() + bar.get_width()/2,
                 (value*100) + 0.005,
                 f"{value*100:.1f}%",fontsize=16,
                 ha='center', va='bottom')
    
    plt.title(plot_title,fontsize=16)
    plt.ylim(0, 50)
    plt.ylabel("Output/Output_sum",fontsize=16)
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{int(x):d}")
    if save==True : 
            plt.savefig(os.path.join(save_plots_path, f"{plot_title}.png"),
                    bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    print(ratio.shape)  # (1024, 1280)
    
    return ratio

Intens = Distribution(save=save_im,save_plots_path=save_plots,
                      plot_title=Flux_distribution)

plt.figure(figsize=(10,5))

def Transmission (line_tr,ref,save,save_plots_path,plot_title):
    line_tr_sum=((np.sum(line_tr,axis=0)/10))
    ref_sum=np.sum(ref,axis=0)
    transmission=(np.divide(line_tr_sum,ref_sum))
    print(transmission,'transmission')
    Tr_tot=np.mean(transmission)
    print(Tr_tot*100)
    plt.plot(axis_x, transmission*100, label='Chip transmission = {:.2f}%'.format(Tr_tot*100))
    plt.axhline(y=25, color='r', linestyle='--',label='25%')
    plt.axis([1075,1450,0,50])
    plt.title(plot_title,fontsize=16)
    plt.ylabel('Transmission[%]',fontsize=16)
    plt.xlabel('Wavelength[nm]',fontsize=16)
    plt.grid(True)
    plt.legend()
    if save==True : 
            plt.savefig(os.path.join(save_plots_path, f"{plot_title}.png"),
                    bbox_inches='tight', pad_inches=0.1)

Transmission(usefull_line,Lines_fiber_alone,save=save_im,save_plots_path=save_plots,
             plot_title=Chip_transmission)



#%%
