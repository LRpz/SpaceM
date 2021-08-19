import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy.spatial import KDTree
from scipy import ndimage
from sf2an.sf2an import load_ss, sf2an, an2sf
import os, glob
import seaborn as sns
import tqdm

def tf_obj(ion_img):
    """Image transformation to apply on ion image for registration.
    Args:
        ion_img (ndarray): the ion image to transform (2D).
    Returns:
        out (array): the transformed ion image.
    """
    # return ion_img
    # return ion_img.T  # --> TF1 HepaJune
    # return np.fliplr(ion_img) #--> TF2 HepaJune, 20171206_CoCulture\M5
    return np.flipud(ion_img) # --> TF for HepaNov17, 20180514_Coculture

def scale(arr):
    return (arr - np.min(arr))/(np.max(arr) - np.min(arr))

def diffusion_scores(df_scores, MF):

    os.chdir(MF + 'Input/MALDI/')
    imzml_name = glob.glob('*.imzML')[0]
    ds_name = imzml_name.replace('.imzML', '')
    hdf5_p = 'C:/Users/rappez\Google Drive\A-Team\projects/1c\co-culture\datasets/all_annotations/2018-08-06-luca-datasets.hdf5'
    df_im0 = pd.read_hdf(hdf5_p)
    df_im = df_im0[df_im0['ds_name'] == ds_name]
    MFA = MF + 'Analysis/'
    marksMask = np.load(MFA + 'Fiducials/transformedMarksMask.npy')
    cellMask = tiff.imread(MFA + 'CellProfilerAnalysis/Labelled_cells.tiff')
    window = 100
    coordX, coordY = np.load(MFA + 'Fiducials/transformedMarks.npy')
    cell_im = plt.imread(MFA + 'CellProfilerAnalysis\Contour_cells_adjusted.png')
    pix_size=0.73

    cellMask_bw_all = cellMask > 0
    pmi = []  # Positive Mark Index
    overLaps = []

    norm_MM = {}
    # Normalized markMask --> express ablation marks coordinates from the stitched global image
    # to the cropped cell image space
    # the coordinates are transformed in integer to be later on used as indexes to generate a mask for each ablation
    # mark over the cropped cell image.

    for mark_ind, data in enumerate(marksMask):
        # print(i)
        norm_MM[str(mark_ind)] = {}
        norm_MM[str(mark_ind)]['x'] = np.array(marksMask[mark_ind, 0] - np.min(coordX) + window).astype(np.int64)
        norm_MM[str(mark_ind)]['y'] = np.array(marksMask[mark_ind, 1] - np.min(coordY) + window).astype(np.int64)

    oleic = tf_obj(np.array(df_im[df_im['mol_formula'] == 'C18H34O2'].image.as_matrix()[0])).ravel()
    # oleic = np.log10(oleic +1)
    colors = plt.cm.jet(scale(oleic))

    dx = ndimage.sobel(cellMask_bw_all, 0)
    dy = ndimage.sobel(cellMask_bw_all, 1)
    edge = np.array(np.sqrt(dx**2 + dy**2).astype(np.float16) > 0, dtype='uint16')
    # plt.imshow(edge)
    x, y = np.where(edge == 1)
    tree = KDTree(list(zip(x,y)))

    # df_scores = pd.DataFrame()


    sfs = ['C23H48NO7P', 'C23H46NO7P']

    # for i, sf in enumerate(df_im['mol_formula'].as_matrix()):
    for i, sf in enumerate(sfs):

        df_scores.loc[i, 'sf'] = sf

        mol_im = tf_obj(np.array(df_im[df_im['mol_formula'] == sf].image.as_matrix()[0]))
        mol_int = mol_im.ravel()
        mol_name = sf2an(sf, ss_df)[0]
        df_scores.loc[i, 'annotation'] = mol_name

        # plt.figure(figsize=(5, 5))
        # ax = plt.subplot(111)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        #
        distS = []
        for mark_ind, data in enumerate(norm_MM):
            AM_x = int(np.mean(norm_MM[str(mark_ind)]['x']))
            AM_y = int(np.mean(norm_MM[str(mark_ind)]['y']))
            dist, index = tree.query((AM_x, AM_y))
        
            if cellMask_bw_all[AM_x, AM_y] == 1:
                fact = -1
            else:
                fact = 1
            xval = dist*fact*pix_size
            mark_intensities = mol_int[int(mark_ind)]
        
            plt.scatter(xval[mark_intensities>0], mark_intensities[mark_intensities>0] , 10, 'k', alpha=0.5)
        
            distS = np.append(distS, xval)
        
        distS = np.clip(distS, np.percentile(distS, 1), np.percentile(distS, 99))
        
        score = np.median(distS[mol_int>0])
        df_scores.loc[i, '{}_score'.format(ds_name[-12:])] = score
        df_scores.loc[i, '{}_mean_I'.format(ds_name[-12:])] = np.mean(mol_int)
        df_scores.loc[i, '{}_mean_I_no0'.format(ds_name[-12:])] = np.mean(mol_int[mol_int>0])
        #
        # plt.axvline(x=0, color=[1,0,0], linewidth=3)
        # plt.title('{}, {}'.format(mol_name, '%.2f' %score), fontsize=15)
        # plt.xlim([-50, 150])
        # plt.ylim([140, 1100])
        # plt.xlabel('Distance from closest cell boundary (um)', fontsize=15)
        # plt.ylabel('Metabolite intensity', fontsize=15)
        #
        # plt.savefig('C:/Users/rappez\Google Drive\A-Team\projects/1c\Paper figures\Supp Figures\co_culture_metabolic_convergence/diffusion/{}.png'.format(sf))
        # plt.savefig('C:/Users/rappez\Google Drive\A-Team\projects/1c\Paper figures\Supp Figures\co_culture_metabolic_convergence/diffusion/{}.svg'.format(sf))
        #
        # plt.close('all')


        plt.figure(figsize=(5, 5))
        plt.imshow(mol_im, cmap='hot')
        plt.title('{}'.format(mol_name), fontsize=15)
        plt.tight_layout()
        plt.savefig('C:/Users/rappez\Google Drive\A-Team\projects/1c\Paper figures\Supp Figures\co_culture_metabolic_convergence/diffusion/{}_img.png'.format(sf))
        plt.close('all')

        # plt.savefig('C:/Users/rappez\Google Drive\A-Team\projects/1c\Paper figures\Supp Figures\co_culture_metabolic_convergence/diffusion\mol_distanceVSintensity/{}.png'.format(sf))
        # plt.close('all')

        # plt.figure(figsize=(5,5))
        # color= [72/255, 133/255, 237/255]
        # sns.kdeplot(distS[mol_int>0], shade=True, alpha=0.2, color=color)
        # sns.kdeplot(distS[mol_int>0], color=color)
        # plt.axvline(x=score, linewidth=2, color=[0,0,0])
        # plt.ylabel('Number of positive ablation mark', fontsize=15)
        # # plt.xlim([-50, 150])
        # plt.xlabel('Distance from closest cell boundary (um)', fontsize=15)
        # plt.title('{}, score={}'.format(mol_name, '%.2f' %score), fontsize=15)
        # plt.tight_layout()
        #
        #
        # plt.savefig('C:/Users/rappez\Google Drive\A-Team\projects/1c\Paper figures\Supp Figures\co_culture_metabolic_convergence/diffusion\mol_distanceVSintensity/{}.png'.format(sf))
        # plt.close('all')

    return df_scores

ss_df = load_ss()
pix_size=0.73
df_scores = pd.DataFrame()
root = 'E:/Experiments/20180514_Coculture_Hela/neg/'
dirs = os.listdir(root)

#Troubleshooting purposes
MF = root+ 'c4_SELECTED' + '/'
diffusion_scores(df_scores, MF)

for dir in tqdm.tqdm(dirs):
    if dir.endswith('SELECTED'):
        MF = root + dir + '/'
        df_scores =  diffusion_scores(df_scores, MF)

all_cols = ['sf', 'annotation']
for col0 in np.array(df_scores.columns)[2:]:
    col1 = col0[6:]
    all_cols = np.append(all_cols, col1)
df_final = pd.DataFrame(data=df_scores.as_matrix(), columns=all_cols)

df_final.to_csv('C:/Users/rappez\Google Drive\A-Team\projects/1c\Paper figures\Supp Figures\co_culture_metabolic_convergence\diffusion\df_scores_all_rep.csv', index=False)






