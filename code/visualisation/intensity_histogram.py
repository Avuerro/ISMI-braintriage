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



def get_acquisition_histogram(in_dir,slices):

    """
        this function will plot the histogram bins for each aquisition 
        for a certain slice FOR ALL PATIENTS of both classes. 
        The histograms of the normal patients will be plotted next to the abnormal patients.
        

        it loops over the slices, sums the intensities and afterwards divides 
        each intensity by the number of patients * the nr of slices
        this way you will have the average intensity of each pixel
    """
    count_t1 = {"normal": [], "abnormal": []}
    count_t2 = {"normal": [], "abnormal": []}
    count_t2_flair = {"normal": [], "abnormal": []}
    
    klasses= ["normal", "abnormal"]
    
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
        normal_data = acquisitions[acquisition]["normal"].flatten()
        normal_max_value = normal_data.max() 
        normal_counts,normal_bins = remove_outliers(normal_data)
        
        abnormal_data = acquisitions[acquisition]["abnormal"].flatten()
        abnormal_max_value = abnormal_data.max() 
        abnormal_counts,abnormal_bins = remove_outliers(abnormal_data)

        max_value = normal_max_value if normal_max_value > abnormal_max_value else abnormal_max_value
        
        # highest_peak = normal_counts[3] if normal_counts[3] > abnormal_counts[3] else abnormal_counts[3]

        ax[i].hist(normal_bins[3:-1], bins = normal_bins, weights=normal_counts[3:],range=(-0.5,normal_max_value), label="normal", alpha=0.5)
        ax[i].hist(abnormal_bins[3:-1], bins = abnormal_bins, weights=abnormal_counts[3:],range=(-0.5,abnormal_max_value), label="abnormal", alpha=0.5)
        ax[i].set_xticks(np.arange(-0.5,max_value, 0.5))

        ax[i].set_title("Average pixel intensities for {} Acquistion \n of patients with and without abnormalities and {} slices".format(acquisition, str(len(slices))))
        ax[i].legend(loc='upper right')
        

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.65,
                    wspace=0.35)
    fig.suptitle("Overview of Average Pixel Intensities \n of the three aquisitions for patiens with and without abnormalities  ".format(klass), y=1.06)
    fig.tight_layout()
    plt.show()


def remove_outliers(data):
    counts,bins = np.histogram(data,bins=256)
    return counts,bins

def get_smallest_value(data):
    smallest_value = np.sort(data)[2:3]

    return smallest_value[0]
