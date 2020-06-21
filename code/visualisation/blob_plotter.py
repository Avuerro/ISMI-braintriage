import numpy as np
import scipy
from scipy import signal,ndimage 
import matplotlib.pyplot as plt
import math

def gaussian_2d(sigma_mm, voxel_size):
    """
    Paramaters
    ----------
    Input:
    sigma_mm: 
    voxel_size: voxel_size
    
    Output:
    kernel: kernel
    x : matrix of x coordinates of the filter
    y : matrix of y coordinates of the filter
    """
    bound = sigma_mm*3
    x,y = np.ogrid[-bound : bound : voxel_size[0], -bound : bound : voxel_size[1]]
    
    kernel = 1/(2*np.pi*sigma_mm**2) * np.exp(-(x**2+y**2)/(2*sigma_mm**2))   
    
    return kernel, x, y 

def laplacian_of_gaussian(g):
    gx, gy = np.gradient(g)
    gxx = np.gradient(gx)[0]
    gyy = np.gradient(gy)[1]
    LoG = gxx + gyy
    return LoG,gxx,gyy


def get_brain(brain_slice):
    """
        The goal of this function is to make the bright areas stand out (more)
        The function blurs the image and rescales it to 0 - 255 
        then it applies a certain treshold, decided upon after trial and error
        finally all values above threshold are set to color white, the rest is black
    
    """
 
    gaussian_blob = gaussian_2d(1, [0.5,0.5])[0]
    brain_slice_blurred = scipy.signal.convolve(brain_slice, gaussian_blob, method="fft", mode="same") 
    # rescaled_brain_slice_blurred =(((brain_slice_blurred - brain_slice_blurred.min()) * (1/(brain_slice_blurred.max() - brain_slice_blurred.min()))) * 255).astype('uint8')
    # variable_threshold = rescaled_brain_slice_blurred.max() * 0.45
    # rescaled_brain_slice_blurred[rescaled_brain_slice_blurred >= variable_threshold ] = 255
    # rescaled_brain_slice_blurred[rescaled_brain_slice_blurred < 200 ] = 0
    rescaled_brain_slice_blurred = brain_slice_blurred
    return rescaled_brain_slice_blurred



def euclidean_dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def notincircle(plotted_circles, newcoords,r):
    for circle in plotted_circles:
        center_coord = circle[0]
        radius = circle[1]
        dist = euclidean_dist(center_coord, newcoords)
        if dist < (radius + r) :
            return False
    return True

def bright_blobs_plotter(image,slices,seed_points=None):

    """
        This function plots blue blobs at the locations with high intensity
        according to the get_brain() function.
    """
    voxel_size = [0.5, 0.5]
    sigmas = [0.5,1,5,50] # We choose 5, 8 and 12 because trachea is 10 to 25mm wide 
                     # and 50 to weaken the effect of large irrelevant areas
    if type(slices) == tuple:
        slices = range(slices[0], slices[1])
    if type(slices) == int:
        slices = [slices]
    for slice_idx in slices:
        ct_slice = image[slice_idx,:,:]
        ct_slice_conv = np.zeros(ct_slice.shape)
        # Compute indices of image that are part of the body
        body_idcs = get_brain(ct_slice)

        # Loop over all sigma values and create average convolved image
        for sigma in sigmas:
            gaussian_blob = gaussian_2d(sigma,voxel_size)[0]
            LoG = laplacian_of_gaussian(gaussian_blob)[0]
            # Use method="same" to avoid size conflict error
            ct_slice_conv += scipy.signal.convolve(ct_slice, LoG,  mode="same")

        ct_slice_conv_avg = ct_slice_conv/float(len(sigmas))
        # Normalize the convolved slice
        ct_slice_conv_avg = (ct_slice_conv_avg - ct_slice_conv_avg.flatten().min())/(ct_slice_conv_avg.flatten().max() - ct_slice_conv_avg.flatten().min())

        # Show original slice
        plt.subplot(1,2,1); plt.imshow(ct_slice, cmap='gray')
        fig = plt.gcf()
        ax = fig.gca()
        ax.set_title("Original slice")

        # Find all indices that are above threshold and part of body
        threshold = 0.15
        max_idcs_y, max_idcs_x = np.where(ct_slice_conv_avg < threshold)
        max_idcs = list(zip(max_idcs_x,max_idcs_y))
        max_idcs_filtered = [idx for idx in max_idcs if body_idcs[idx[0],idx[1]]]
        max_idcs_filtered = sorted(max_idcs_filtered)

        ## We need to create groups, we loop through the coordinates and compare them to eachother
        ## If the difference is smaller than 7 we consider the two points a group.
        groups =[[]]
        group_number = 0
        for p1,p2 in zip(max_idcs_filtered[:-1],max_idcs_filtered[1:]):
            if (euclidean_dist(p1,p2) < 7):
                groups[group_number].append(p1)
                groups[group_number].append(p2)
            else:
                group_number += 1
                groups.append([])
        groups = [x for x in groups if x != []]
        groups = sorted(groups, key = len, reverse=True)

        ## now we would like to draw a box around each group
        ## we need to find max x difference and max y difference
        ## the biggest will be the radius of the circle

        groups_coords = []
        for group in groups:
            group = sorted(group)
            p1 = group[0]
            p2 = group[-1]
            x_difference = np.abs(p1[0]-p2[0])
            y_difference = np.abs(p1[1]-p2[1])

            radius = x_difference if x_difference > y_difference else y_difference
            center_point = group[len(group) // 2]

            groups_coords.append([center_point,radius])

        plt.subplot(1,2,2); plt.imshow(ct_slice, cmap='gray')
        fig = plt.gcf()
        ax = fig.gca()
        ax.set_title("Original slice with red circles around bright spots")
        plotted_circles = []
        for index, group in enumerate(groups_coords):
            center = group[0]
            radius = group[1]
            if (radius < 20):
                radius += 10
            if ( notincircle(plotted_circles, center,radius) or index == 0):
                ax.add_artist(plt.Circle(center,radius,color="red", fill = False, linewidth=2))
                plotted_circles.append([center, radius ])
        plt.show()
        


