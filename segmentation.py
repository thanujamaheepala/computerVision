import cv2 as cv
import numpy as np
import os
import math

def get_images_from_folder(folder):
    imagesNames = []
    images = []
    for filename in os.listdir(folder):
        if (filename.endswith(".jpeg")|filename.endswith(".jpg")):
            img = cv.imread(os.path.join(folder,filename))
            if img is not None:
                imagesNames.append(filename)
                images.append(img)
    return imagesNames,images

def rgbToGreyScale(image): #convert RGB to GreyScale   
    greyScaleImage = []  
    for i in range(len(image)):
        result_row = []
        for j in range(len(image[0])):
            greyScale  = int((0.299*image[i][j][2]) + (0.587*image[i][j][1]) + (0.114*image[i][j][0]))
            result_row.append(greyScale)
        greyScaleImage.append(result_row) 
    return greyScaleImage

def getHistogram(image):  #get histogram of a Image
    histogram = {}
    for n in range(256):
        histogram[n] = 0   
    for i in range(len(image)):
        for j in range(len(image[0])):
            histogram[image[i][j]]+=1 
    return histogram

def getSmoothedHistogram(histogram):  #Smooth histogram with width 3
    keys = list(histogram.keys())
    l = len(keys) 
    for i in range(l):
        if(i==0):
            newValue=int((histogram[keys[i]]+histogram[keys[i+1]])/2)
        elif(i==(l-1)):
            newValue=int((histogram[keys[i-1]]+histogram[keys[i]])/2)
        else:
            newValue=int((histogram[keys[i-1]]+histogram[keys[i]]+histogram[keys[i+1]])/3)
        histogram[keys[i]] = newValue
    return histogram

def getSegmentedImage(image,threshold): # segment image using threshold
    for i in range(len(image)):
        for j in range(len(image[0])):
            if((image[i][j])<=threshold):
                image[i][j] = 255
            else:
                image[i][j] = 0
    return image

def getAvgIntensity(histogram): #find avg intensity of a histogram
    totalIntensigty = 0
    for key in histogram.keys():
        totalIntensigty+=histogram[key]*key

    return int(totalIntensigty/sum(histogram.values()))

def getThresholdedHistograms(histogram, threshold): #find lower and higher thresholded histograms
    lowerHistogram = {}
    upperHistogram = {}
    for key in histogram.keys():
        if(key<=threshold):
            lowerHistogram[key] = histogram[key]
        else:
            upperHistogram[key] = histogram[key]

    return lowerHistogram,upperHistogram

def interMeansAlgorithm(histogram,thresholds=[]): #find threshold using inter-mean algorithm
    if(len(thresholds)==0):
        thresholds = [getAvgIntensity(histogram)]
    lowerHistogram,upperHistogram = getThresholdedHistograms(histogram, thresholds[-1])
    lowerAvgIntensity = getAvgIntensity(lowerHistogram)
    upperAvgIntensity = getAvgIntensity(upperHistogram)
    newThreshold = int((lowerAvgIntensity+upperAvgIntensity)/2)
    if(newThreshold in thresholds[-2:]):
        return newThreshold
    else:
        thresholds.append(newThreshold)
        return interMeansAlgorithm(histogram, thresholds)


#get all images and image names in the current directory
imagesNames,images = get_images_from_folder((os.path.dirname(os.path.abspath(__file__))))


for i in range(len(images)):
    
    # convert RGB to greyScale
    greyImage = rgbToGreyScale(images[i]) 

    # get histogram of the image
    histogram = getHistogram(greyImage)

    #width 3 smoothing
    smoothedHistogram = getSmoothedHistogram(histogram)

    # find a threshold using inter means algorithm
    interMeansThreshold = interMeansAlgorithm(smoothedHistogram)

    # segement the image using threshold (Inter-mean)
    interMeansSegmentedImage = getSegmentedImage(greyImage,interMeansThreshold) 

    # saving the image
    [imageName,extension] = imagesNames[i].split(".")
    cv.imwrite(imageName+"_segmented."+extension, np.array(interMeansSegmentedImage))

    

