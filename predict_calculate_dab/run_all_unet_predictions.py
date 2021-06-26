# This file runs through all of the three unets that were made for the project:
#     one for acini, nuclei, and stroma
# Then the output images are masked - so you get 5 separete images from each input TA image
# Finally,  the amount of DAB staining in these images is calculated.


import os
import numpy as np
import skimage.transform as trans
from cnncore.corepredict import tile_single_core, fit_prediction  # CA
from cnncore.predict_unet import predict_on_core, predict_single_image  # CA
import dab.dab_analysis as dab  # CA
from skimage.transform import rescale
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from PIL import Image
import xml.etree.ElementTree as ET # metadata xml handling
import numpy as np
from skimage.draw import polygon
import imageio # saving mask
from skimage import img_as_ubyte #saving mask
import matplotlib
from openslide import OpenSlide
from skimage import filters, color, measure, morphology
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
from PIL import Image
from datetime import datetime
from keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

imagepath = "./images/"
outpath = "./OUTPUT/"
modelpath = "./MODELS/"


# Run all 3 models in turn
def save_core_predictions(modelname, imagepath="./images/", outpath="./OUTPUT/"):
    """
    use the tmalib library to predict on a whole core and save results given a model name
    :param model: the name of the model - "acini", "nuclei", "tissue"
    :param imagepath: path to the images
    :param outpath: path to the whole output directory = "./OUTPUT/
    :return: None
    """
    files = sorted([f for f in os.listdir(imagepath) if f.endswith('.png')])
    modelpath = "./MODELS/unet_model_" + modelname + ".hdf5"

    # load the model
    model = load_model(modelpath)
    #print("loaded model\nmaking predictions")
    # run through images and save presictions
    for i in range(len(files)):
        print(files[i], modelname, f"{i}/{len(files)}")
        try:
            if modelname == "tissue":
                precore = predict_single_image(model, imagepath + files[i], target_size=(256, 256))
            else:
                precore = predict_on_core(model, imagepath=imagepath+files[i], overlap=600)
            Image.fromarray(precore).save(outpath + modelname + "_predictions" + os.sep + files[i])
            # mpimg.imsave(outpath + modelname + "_predictions" + os.sep + files[i], precore)
            #print("done image - " + files[i] + " prediction - " + modelname)
        except Exception as e:
            print(e, "ERROR")
            continue


print('----creating prediction maps----')
for m in ["acini", "nuclei", "tissue"]:
    save_core_predictions(modelname=m, imagepath="./images/", outpath="./OUTPUT/")


def segmentation_images(imagepath="./images/", outpath="./OUTPUT/"):
    """
    Run through all images and predictions to fill 5 directories with segmentations
    needs to have the file structure labelled above
    """
    files = sorted([f for f in os.listdir(imagepath) if f.endswith('.png')])
    for i in range(len(files)):
        image = mpimg.imread(imagepath + files[i])[:, :, :3]
        print(files[i], f"{i}/{len(files)}")

        # nuclei
        nuclei = mpimg.imread(outpath + "nuclei_predictions/" + files[i])
        nuclei = color.rgb2gray(nuclei)
        nuclei = nuclei > filters.threshold_otsu(nuclei)
        nuclei_only = np.copy(image)
        nuclei_only[nuclei == False] = 0
        mpimg.imsave(outpath + "nuclei/" + files[i], nuclei_only)
        del nuclei_only
        #         print("nuclei saved")

        # acini
        acini = mpimg.imread(outpath + "acini_predictions/" + files[i])
        acini = color.rgb2gray(acini)
        acini = acini > filters.threshold_otsu(acini)
        acini_only = np.copy(image)
        acini_only[acini == False] = 0
        mpimg.imsave(outpath + "acini/" + files[i], acini_only)
        del acini_only
        #         print("acini saved")

        # acinar_nuclei
        acinar_nuclei = np.logical_and(acini == True, nuclei == True)
        acinar_nuclei_only = np.copy(image)
        acinar_nuclei_only[acinar_nuclei == False] = 0
        mpimg.imsave(outpath + "acinar_nuclei/" + files[i], acinar_nuclei_only)
        #         print("acinar_nuclei saved")
        del acinar_nuclei
        del acinar_nuclei_only

        # stromal_nuclei
        stromal_nuclei = np.logical_and(acini == False, nuclei == True)
        stromal_nuclei_only = np.copy(image)
        stromal_nuclei_only[stromal_nuclei == False] = 0
        mpimg.imsave(outpath + "stromal_nuclei/" + files[i], stromal_nuclei_only)
        #         print("stromal_nuclei saved")
        del stromal_nuclei
        del stromal_nuclei_only

        # core - not saved, just for stroma calculation
        core = mpimg.imread(outpath + "tissue_predictions/" + files[i])
        core = color.rgb2gray(core)
        core = rescale(core, 23.4375, multichannel=False)  # scale factor is 6000/256 = 23.4375
        core = core > filters.threshold_mean(core)  # threshold the prediction

        # stroma
        stroma_only = np.copy(image)
        stroma_only[core == False] = 0
        del core
        stroma_only[acini == True] = 0  # removes acini from the mask
        mpimg.imsave(outpath + "stroma/" + files[i], stroma_only)
        #         print("acinar_nuclei saved")
        del stroma_only
    print("all segmentation done")


print('----creating segmentation images from predictions----')
segmentation_images(imagepath="./images/", outpath="./OUTPUT/")


# run dab analysis on the segmentation images
def update_df(df, dirname, filename, coverage, mean, std_dev, segmented_pixel_count):
    """
    update the coverage, mean, std_dev in a given df for a specific segmentation type (dirname)
    """
    df.loc[df['names'] == filename, f'{dirname}_coverage'] = coverage
    df.loc[df['names'] == filename, f'{dirname}_mean'] = mean
    df.loc[df['names'] == filename, f'{dirname}_std_dev'] = std_dev
    df.loc[df['names'] == filename, f'{dirname}_segmentation_count'] = segmented_pixel_count 
    return df


def get_amt(outpath, filename):
    """
    calculate amount of tissue
    """
    core = mpimg.imread(outpath + "tissue_predictions" + os.sep + filename)[:, :, :3]
    core = color.rgb2gray(core)
    core = rescale(core, 23.4375, multichannel=False)  # scale factor is 6000/256 = 23.4375
    core = core > filters.threshold_mean(core)  # threshold the prediction
    amt = np.count_nonzero(core)
    del core
    return amt


def analyse_dab_segmentation(imagepath, outpath):
    """
    run through all segmentations for each image/ calculate dab / save all results
    """
    print("running dab analysis")
    files = sorted([f for f in os.listdir(imagepath) if f.endswith('.png')])
    # Initialise results dataframe
    df = pd.DataFrame(columns=["names", "acini_coverage", "acini_mean", "acini_std_dev", "acini_segmentation_count",
        "nuclei_coverage", "nuclei_mean", "nuclei_std_dev","nuclei_segmentation_count",
        "stroma_coverage", "stroma_mean", "stroma_std_dev", "stroma_segmentation_count",
        "stromal_nuclei_coverage", "stromal_nuclei_mean", "stromal_nuclei_std_dev",  "stromal_nuclei_segmentation_count",
        "acinar_nuclei_coverage", "acinar_nuclei_mean", "acinar_nuclei_std_dev", "acinar_nuclei_segmentation_count", 
        "AMT"])
    df['names'] = files

    dirnames = ['acini', 'nuclei', 'stroma', 'stromal_nuclei', 'acinar_nuclei']
    for i in range(len(files)):
        for dirname in dirnames:
            print(files[i], dirname, f"{i}/{len(files)}")
            image = mpimg.imread(outpath + dirname + os.sep + files[i])[:, :, :3]
            coverage, mean, std_dev, segmented_pixel_count = dab.stain_quantify(image, value=50, segmented_image=True, save=None)  # quantify dab in images
            df = update_df(df, dirname, files[i], coverage, mean, std_dev, segmented_pixel_count)

        # get the AMT
        print(files[i], "AMT", f"{i}/{len(files)}")
        amt = get_amt(outpath, files[i])
        df.loc[df['names'] == files[i], 'AMT'] = amt

    # save results
    date = datetime.now().strftime('%d-%m-%y__%H_%M')
    df.to_excel(outpath + "result_spreadsheets/DAB_SEGMENTATION_ANALYSIS" + date + ".xlsx", index=False)
    print("analysis complete: results saved")
    return df

print('----analysing dab segmentation images----')
analyse_dab_segmentation(imagepath, outpath)
# analyse_dab_segmentation("./origional_JLTA_AA51/", "./JLTA_AA51_OUTPUT/")



