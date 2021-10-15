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

def rgbToGreyScale(image):
    
    greyScaleImage = []
    
    for i in range(len(image)):
        result_row = []
        for j in range(len(image[0])):
            greyScale  = int((0.299*image[i][j][2]) + (0.587*image[i][j][1]) + (0.114*image[i][j][0]))
            result_row.append(greyScale)
        greyScaleImage.append(result_row)
    
    return greyScaleImage

def getGaussianFilteredArray(imageArray):
    filterArray = [[1,2,3,2,1],[2,7,11,7,2],[3,11,17,11,3],[2,7,11,7,2],[1,2,3,2,1]]
    divider=121
    f=len(filterArray)
    N = len(imageArray)
    M = len(imageArray[0])
    result_array = []
    for x in range(N-(f-1)):
        result_row = []
        for y in range(M-(f-1)):
            result = 0
            for i in range(x,x+f):
                for j in range (y,y+f):
                    result+= imageArray[i][j]*filterArray[i-x][j-y]      
            result_row.append(int(result/divider))
        result_array.append(result_row)
    return result_array

def getGradient(imageArray):
    # using sobel filter
    filterX = [[-1,0,1],[-2,0,2],[-1,0,1]]
    filterY = [[1,2,1],[0,0,0],[-1,-2,-1]]
    f=len(filterX)
    N = len(imageArray)
    M = len(imageArray[0])
    g = []
    d = []
    for x in range(N-(f-1)):
        resultRow = []
        resultRowDirection = []
        for y in range(M-(f-1)):
            resultX = 0
            resultY = 0
            for i in range(x,x+f):
                for j in range (y,y+f):
                    resultX += imageArray[i][j]*filterX[i-x][j-y]  
                    resultY += imageArray[i][j]*filterY[i-x][j-y] 
            resultRow.append(int((resultX**2 + resultY**2)**0.5))
            direction = round(math.degrees(math.atan2(resultY,resultX)) / 45) * 45
            if (direction<0):
                direction = 180+direction
            if (direction==180):
                direction = 0
            resultRowDirection.append(direction)
        g.append(resultRow)
        d.append(resultRowDirection)
    return g,d

def nonMaximaSuppression(strengthArray,directionArray):
    f=3
    N = len(strengthArray)
    M = len(strengthArray[0])
    result_array = []
    for x in range(N-(f-1)):
        result_row = []
        for y in range(M-(f-1)):
            resultArrayS = []
            resultArrayD = []
            for i in range(x,x+f):
                for j in range (y,y+f):
                    resultArrayS.append(strengthArray[i][j])
                    resultArrayD.append(directionArray[i][j])
            placesToCompare = {
                0: [3,5],
                45: [2,6],
                90: [1,7],
                135:[0,8]
                }
            maxNeighborStrength = 0
            for i in placesToCompare[directionArray[x+1][y+1]]:
                if (directionArray[x+1][y+1]==resultArrayD[i]):
                    if(resultArrayS[i]>maxNeighborStrength):
                        maxNeighborStrength = resultArrayS[i]
            res = 0
            if (strengthArray[x+1][y+1]>=maxNeighborStrength):
                res = strengthArray[x+1][y+1]
            result_row.append(int(res))
        result_array.append(result_row)
    return result_array

def doubleThresholding(imageArray,upper,lower):
    f=3
    N = len(imageArray)
    M = len(imageArray[0])
    result_array = []
    for x in range(N-(f-1)):
        result_row = []
        for y in range(M-(f-1)):
            resultArray = []
            for i in range(x,x+f):
                for j in range (y,y+f):
                    if (imageArray[i][j]>=upper):
                        resultArray.append(imageArray[i][j])    
            res = 0
            if (imageArray[x][y]>=upper):
                res = imageArray[x][y]
            else:
                if ((imageArray[x][y]>=lower)&(len(result_array)!=0)):
                    res = 255
            result_row.append(int(res))
        result_array.append(result_row)
    return result_array


#get all images and image names in the current directory
imagesNames,images = get_images_from_folder((os.path.dirname(os.path.abspath(__file__))))


for i in range(len(images)):
    
    # convert RGB to greyScale
    resultImage = rgbToGreyScale(images[i]) 
    
    # Filter using gaussian filter
    resultImage = getGaussianFilteredArray(resultImage) 

    # get gradient strength array and direction array with sobel filter
    strengthArray,directionArray = getGradient(resultImage) 
    
    sobelImage = np.array(strengthArray) 

    # do non maxima suppression
    resultImage = nonMaximaSuppression(strengthArray,directionArray) 

    cannyImageBeforeThreshold = np.array(resultImage)

    # dual threshold
    resultImage = doubleThresholding(resultImage,255,80)

    cannyImageAfterThreshold = np.array(resultImage)

    # saving the image
    [imageName,extension] = imagesNames[i].split(".")
    cv.imwrite(imageName+"_sobel."+extension, sobelImage)
    cv.imwrite(imageName+"_canny."+extension, cannyImageAfterThreshold)
    cv.imwrite(imageName+"_canny_before_threshold."+extension, cannyImageBeforeThreshold)
    

