from spaceM.Pipeline import getPath, stitchMicroscopy, \
    ablationMarksFinder, \
    fiducialsFinder, \
    ablationMarksFilter, \
    registration, \
    cellSegmentation, \
    spatioMolecularMatrix, \
    curator
import numpy as np

def img_tf(img):
    """Image transformation to apply to the tile images prior to stitching
    Args:
        img (array): the image to transform (2D).
    Returns:
        out (array): the transformed ion image.
    """
    # return img #20171206_CoCulture
    return np.fliplr(img) #20171106_Hepa_Nov, 20170622_Hepa_June_DAN_Untreated_FA_LPS_TNFa

def ion2fluoTF(ion_img):
    """Image transformation to apply on ion image for registration.
    Args:
        ion_img (ndarray): the ion image to transform (2D).
    Returns:
        out (array): the transformed ion image.
    """
    return ion_img.T  # --> TF1 HepaJune dataset batches FASTER
    # return np.fliplr(ion_img) #--> TF2 HepaJune, 20171206_CoCulture\M5
    # return np.flipud(ion_img) # --> TF for HepaNov17

channels = ['gray', 'red', 'green', 'blue']

MF = getPath('MF')
print('Main folder: {}'.format(MF))

stitchMicroscopy(MF,
                 preMALDI=True,
                 postMALDI=False,
                 tf=img_tf,
                 merge_colors=[],
                 merge_filenames=[])
ablationMarksFinder(MF)
fiducialsFinder(MF)

curator()

ablationMarksFilter(MF)
registration(MF, tf_obj=ion2fluoTF, ili_fdr=0.2)

cellSegmentation(MF, merge_colors=['gray', 'red'], merge_filenames=['img_t1_z1_c1.tif', 'img_t1_z1_c2.tif'])
spatioMolecularMatrix(MF, tf_obj=ion2fluoTF)

