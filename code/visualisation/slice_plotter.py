import torch
import matplotlib.pyplot as plt


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

        fig,axs = plt.subplots(nr_of_rows,nr_of_cols, figsize=(16,16))


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
    nr_of_cols = row_col_number # max 4 plots per row 
    return size_of_range, nr_of_rows, nr_of_cols, subplots_to_delete ,slices