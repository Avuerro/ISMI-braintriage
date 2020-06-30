import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import numpy as np

def compare_slices(normal,abnormal,range_of_slices,DATA_DIR):
    ## for each slice 3 rows and 2  cols

    if ( type(range_of_slices) == int ):
        range_of_slices =[range_of_slices]
    
    nr_of_rows = len(range_of_slices) * 3
    nr_of_cols = 2

    fig,axs = plt.subplots(nr_of_rows,nr_of_cols, figsize=(16,40), squeeze=False)

    
    klasses = ["normal", "abnormal"]
    acquisitions = ["T1", "T2", "T2-FLAIR"]


    for i, patient in enumerate([normal,abnormal]):
        for j,slice_number in enumerate(range_of_slices):
            for k, acquisiton in enumerate(acquisitions):   
                # load the mha files
                acquisiton_path = os.path.join(DATA_DIR, klasses[i], patient,'{}.mha'.format(acquisiton))
                ## Load all images as numpy arrays
                acquisition_array = sitk.GetArrayFromImage(sitk.ReadImage(acquisiton_path))
                acquisition_slice = acquisition_array[slice_number,:,:]

                axs[k + (j * 3)][i].imshow(acquisition_slice, cmap="gray")
                axs[k + (j * 3)][i].set_title("{} Patient {} slice number {} \n Acquistion {} ".format(klasses[i],patient,slice_number, acquisiton))

    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.65, wspace=0.35)
    fig.tight_layout()

    
def plot_patient_slices(patient_slices):
    """
        This functions plots the T2-Flair images of 1 patient, given all slices

        Params:
            patient_slices: all slices of a patient (shape: [32, 3, 426, 426])

    """
    t2_flair_slices = patient_slices[:,2,:,:]
    fig, axs = plt.subplots(8,4, figsize=(30, 30))
    fig.subplots_adjust(hspace = .2, wspace=0.2, left=0.5)
    axs = axs.ravel()
    for i in range(len(axs)):
        axs[i].set_title("T2_Flair Slice Number {}".format(i))
        axs[i].axis('off')
        axs[i].imshow(t2_flair_slices[i], cmap='gray')
        

def plot_slices(patient,range_of_slices,DATA_DIR,row_col_number):
    """
        This functions plots a single slice or all slices in a list or tuple.

        Params: 
            patient: integer indicating the patient number
            range_of_slices: integer slice or list or tuple of slices you want to plot. 
            DATA_DIR : location of the data
            row_col_number: amount that dictates the amount of colomns and rows.

    """
    if (type(range_of_slices) == tuple)  or (type(range_of_slices) == list) :

        size_of_range, nr_of_rows, nr_of_cols, subplots_to_delete ,slices = plot_slices_helper(range_of_slices,row_col_number) 

        fig,axs = plt.subplots(nr_of_rows,nr_of_cols,  figsize=(18,12))


        for index,slice_number in enumerate(slices): #inclusive..
            X = torch.load(DATA_DIR + '/' + str(patient) + '_' + str(slice_number) + '.pt')
            if nr_of_rows > 1:
                axes = axs[ index//row_col_number , index%row_col_number ]
            else:
                axes = axs[index]
            axes.imshow(X.data.numpy()[2,:,:],cmap="gray")
            axes.set_title("Patient {} slice number {}".format(patient,slice_number))
            
        # make sure we don't have these empty white subplots
        for empty_subplot in range(row_col_number - 1 , row_col_number - subplots_to_delete - 1, -1):
            fig.delaxes(axs[nr_of_rows-1][empty_subplot])
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.65,
                    wspace=0.35)
        fig.tight_layout()
    else:
        slice_number = range_of_slices
        X=torch.load(DATA_DIR + '/' + str(patient) + '_' + str(slice_number) + '.pt')

        plt.figure()
        plt.imshow(X.data.numpy()[2,:,:]) 
        plt.title("Patient {} slice number {}".format(patient,slice_number))
    plt.show()

def get_acquisitions(patient,klass, slice_number,DATA_DIR):
    patient = str(patient)
    t1_path = os.path.join(DATA_DIR, klass, patient,'T1.mha')
    t2_path = os.path.join(DATA_DIR, klass, patient,'T2.mha')
    t2_flair_path = os.path.join(DATA_DIR, klass, patient,'T2-FLAIR.mha')

    ## Load all images as numpy arrays
    t1_array = sitk.GetArrayFromImage(sitk.ReadImage(t1_path))
    t2_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_path))
    t2_flair_array = sitk.GetArrayFromImage(sitk.ReadImage(t2_flair_path))

    t1_slice = t1_array[slice_number,:,:]
    t2_slice = t2_array[slice_number,:,:]
    t2_flair_slice = t2_flair_array[slice_number,:,:]
    comb_data = np.array([t1_slice, t2_slice, t2_flair_slice])
    return comb_data

def plot_slices_by_acquisition(patient,klass,range_of_slices,DATA_DIR):

    """
        This function plots the acquisitions for each slice by calling 
        the plot_slice_by_acquisition method which plots the aqcuisitions for each slice

    """
    if(type(range_of_slices) == int):
        slice_number = range_of_slices
        X = get_acquisitions(patient,klass,slice_number,DATA_DIR)#torch.load(DATA_DIR + '/' + str(patient) + '_' + str(slice_number) + '.pt')
        plot_slice_by_acquisition(X, slice_number)
    else:
        size_of_range, nr_of_rows, nr_of_cols, subplots_to_delete ,slices = plot_slices_helper(range_of_slices,3) # cols are always three
        for index,slice_number in enumerate(slices): #inclusive..
            X = get_acquisitions(patient,klass,slice_number,DATA_DIR)#torch.load(DATA_DIR + '/' + str(patient) + '_' + str(slice_number) + '.pt')
            plot_slice_by_acquisition(X, slice_number)

def plot_slice_by_acquisition(data,slice_number):
    """
        This function plots the acquisitions for a single slice.
    """
    fig,axs = plt.subplots(1,3, figsize=(18,12))
    acquisitions = ['T1','T2', 'T2_flair']
    for index,acquisition in enumerate(acquisitions):
        axs[index].imshow(data[index], cmap="gray")
        axs[index].set_title("Acquisition {}".format(acquisition))
        axs[index].grid(False)
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.65,
                    wspace=0.35)
    fig.suptitle('Slice {}'.format(slice_number), fontsize=12, x=0, y = 0.5,ha='left' )


def plot_slices_helper(range_of_slices,row_col_number):

    """
        Helper function that takes over general functionality for the plot_slices function
        
        Params:
            range_of_slices: integer slice or list or tuple of slices you want to plot. 
            row_col_number: amount that dictates the amount of colomns and rows.
    """
    
    if type(range_of_slices) == tuple:
        assert range_of_slices[0] < range_of_slices[1]
        size_of_range = range_of_slices[1] - range_of_slices[0]

        subplots_to_delete = (row_col_number - size_of_range%row_col_number) - 1 
        slices = range(range_of_slices[0],range_of_slices[1]+1)
    
    if type(range_of_slices) == list:
        size_of_range = len(range_of_slices)

        subplots_to_delete = (row_col_number - size_of_range%row_col_number)

        slices = range_of_slices

    nr_of_rows = size_of_range // row_col_number + 1
    nr_of_cols = row_col_number  
    return size_of_range, nr_of_rows, nr_of_cols, subplots_to_delete ,slices