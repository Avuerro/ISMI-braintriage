import os
import numpy as np
import SimpleITK as sitk

import matplotlib.pyplot as plt

##normalize mri image based on
#https://github.com/jcreinhold/intensity-normalization/blob/master/intensity_normalization/normalize/zscore.py

def zscore_normalize(img):
    """
        Normalization of image based on mean and std.
    """
    mean = img.mean()
    std = img.std()
    normalized_image = (img - mean) / std
    return normalized_image



def get_acquisition_histogram(in_dir,slices, klass=None):

    """
        this function will plot the histogram bins for each aquisition 
        for a certain slice FOR ALL PATIENTS of both classes or of the specified class. 
        If no class parameter is provided then the Histograms for both normal and abnormal patients
        will be plotted in an overlapping manner.

        it loops over the slices, sums the intensities and afterwards divides 
        each intensity by the number of patients * the nr of slices
        this way you will have the average intensity of each pixel
    """
    count_t1 = {"normal": [], "abnormal": []}
    count_t2 = {"normal": [], "abnormal": []}
    count_t2_flair = {"normal": [], "abnormal": []}

    if (type(klass) == str): klass = [klass] # klass should be a list

    if (klass == None or len(klass)>1 ):
        klasses= ["normal", "abnormal"]
    else:
        klasses = klass
    
    acquisitions = {"t1":count_t1, "t2": count_t2, "t2_flair": count_t2_flair}

    for klass in klasses:

        patients_dir = os.path.join(in_dir, klass)
        patients = os.listdir(patients_dir)
        
        if type(slices) == tuple:
            slices = range(slices[0], slices[1])
        if type(slices) == int:
            slices = [slices]
        for index,patient in enumerate(patients):
            patient = str(patient)
            t1_path = os.path.join(in_dir, klass, patient,'T1.mha')
            t2_path = os.path.join(in_dir, klass, patient,'T2.mha')
            t2_flair_path = os.path.join(in_dir, klass, patient,'T2-FLAIR.mha')

            ## Load all images as numpy arrays
            t1_array = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
            t2_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
            t2_flair_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_flair_path))

            for slice_number in slices:
                ## obtain the slices
                t1_slice = t1_array[slice_number,:,:]
                t2_slice = t2_array[slice_number,:,:]
                t2_flair_slice = t2_flair_array[slice_number,:,:]


                ## normalize the images
                t1_slice_normalized = zscore_normalize(t1_slice)
                t2_slice_normalized = zscore_normalize(t2_slice)
                t2_flair_slice_normalized = zscore_normalize(t2_flair_slice)

                if index == 0 :
                    count_t1[klass] = t1_slice_normalized
                    count_t2[klass] = t2_slice_normalized
                    count_t2_flair[klass] = t2_flair_slice_normalized
                else:
                    count_t1[klass] += t1_slice_normalized
                    count_t2[klass] += t2_slice_normalized
                    count_t2_flair[klass] += t2_flair_slice_normalized  
            
        
        count_t1[klass] = count_t1[klass] / (len(patients) * len(slices))
        count_t2[klass] = count_t2[klass] / (len(patients) * len(slices))
        count_t2_flair[klass] = count_t2_flair[klass] / (len(patients) *len(slices))
    ## 3 aquisitions for each class : 3 * 2 = 6
    ## 2 aquisitions  per row
    ## 3 rows , 2 columns
    fig,ax = plt.subplots(3,1, figsize=(14,14))


    for i,acquisition in enumerate(acquisitions.keys()):
        # for j,klass in  enumerate(acquisitions[acquisition].keys()):
        max_value = 0
        alpha_value = 1 / len(klasses)
        for klass in klasses:
            data  = acquisitions[acquisition][klass].flatten()    
            data_max_value = data.max()
            data_counts, data_bins = remove_outliers(data)
            if ( len(klasses)> 1 ):
                max_value = max(data_max_value,max_value)
            else:
                max_value = data_max_value
            

            ax[i].hist(data_bins[3:-1], bins = data_bins, weights=data_counts[3:],range=(-0.5,max_value), label=klass, alpha=alpha_value)
    
        ax[i].set_xticks(np.arange(-0.5,max_value, 0.5))
        if len(klasses)>1:
            ax[i].set_title("Average pixel intensities for {} Acquistion \n of patients with and without abnormalities and {} slices".format(acquisition, str(len(slices))))
            ax[i].legend(loc='upper right')
        else:
            ax[i].set_title("Average pixel intensities for {} Acquistion \n of {} patients  and {} slices".format(acquisition, klasses[0], str(len(slices))))
            # ax[i].legend(loc='upper right')            
        

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.65,
                    wspace=0.35)
    if (len(klasses) > 1 ):                
        fig.suptitle("Overview of Average Pixel Intensities \n of the three aquisitions for patiens with and without abnormalities  ", y=1.06)
    else:
        fig.suptitle("Overview of Average Pixel Intensities \n of the three aquisitions for {} patiens".format(klass), y=1.06)
    fig.tight_layout()
    plt.show()


def remove_outliers(data):
    counts,bins = np.histogram(data,bins=256)
    return counts,bins

def get_smallest_value(data):
    smallest_value = np.sort(data)[2:3]

    return smallest_value[0]
