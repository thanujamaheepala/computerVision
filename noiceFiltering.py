import cv2 as cv
import numpy as np
import os

def getWrappedImage(imageArray):
    imageArray = [imageArray[-1]]+imageArray+[imageArray[0]]
    wrappedImageArray = []
    for row in imageArray:
        wrappedImageArray.append([row[-1]]+row+[row[0]])   
    return wrappedImageArray

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

def getMeanFilteredArray(imageArray,filterArray,divider=1):
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

def getMedianFilteredArray(imageArray,filterArray):
    f=len(filterArray)
    N = len(imageArray)
    M = len(imageArray[0])
    result_array = []
    for x in range(N-(f-1)):
        result_row = []
        for y in range(M-(f-1)):
            resultArray = []
            for i in range(x,x+f):
                for j in range (y,y+f):
                    resultArray.append(imageArray[i][j])    
            resultArray.sort()
            mid = len(resultArray) // 2
            res = (int(resultArray[mid]) + int(resultArray[~mid]))/ 2
            result_row.append(int(res))
        result_array.append(result_row)
    return result_array

def getMidPointFilteredArray(imageArray,filterArray):
    f=len(filterArray)
    N = len(imageArray)
    M = len(imageArray[0])
    result_array = []
    for x in range(N-(f-1)):
        result_row = []
        for y in range(M-(f-1)):
            resultArray = []
            for i in range(x,x+f):
                for j in range (y,y+f):
                    resultArray.append(imageArray[i][j])    
            resultArray.sort()
            mid = len(resultArray) // 2
            res = (int(resultArray[0]) + int(resultArray[-1]))/ 2
            result_row.append(int(res))
        result_array.append(result_row)
    return result_array

def getFilter(N):
    divider = N*N
    result_array = []
    for i in range(N):
        result_row = []
        for j in range(N):
            result_row.append(1)
        result_array.append(result_row)
    return divider, result_array

def splitImage(image):
    b = []
    for i in range(len(image)):
        result_row = []
        for j in range(len(image[0])):
            result_row.append(image[i][j][0])
        b.append(result_row)
    g = []
    for i in range(len(image)):
        result_row = []
        for j in range(len(image[1])):
            result_row.append(image[i][j][1])
        g.append(result_row)
    r = []
    for i in range(len(image)):
        result_row = []
        for j in range(len(image[2])):
            result_row.append(image[i][j][2])
        r.append(result_row)

    return b,g,r

def mergeImage(b,g,r):
    
    filteredImage = []

    for i in range(len(b)):
        result_row = []
        for j in range(len(b[0])):
            result_row.append([b[i][j],g[i][j],r[i][j]])
        filteredImage.append(result_row)

    return np.array(filteredImage)

imagesNames,images = get_images_from_folder((os.path.dirname(os.path.abspath(__file__))))

meanFilteredImages = []
medianFilteredImages = []
midPointFilteredImages = []


divider,filterArray=getFilter(3)

for image in images:

    b,g,r = splitImage(image)

    b,g,r = getWrappedImage(b),getWrappedImage(g),getWrappedImage(r)

    b1,g1,r1 = getMedianFilteredArray(b,filterArray), getMedianFilteredArray(g,filterArray), getMedianFilteredArray(r,filterArray)

    filteredImage1 = mergeImage(b1,g1,r1)

    medianFilteredImages.append(filteredImage1)

    b2,g2,r2 = getMeanFilteredArray(b,filterArray,divider), getMeanFilteredArray(g,filterArray,divider), getMeanFilteredArray(r,filterArray,divider)

    filteredImage2 = mergeImage(b2,g2,r2)

    meanFilteredImages.append(filteredImage2)

    b3,g3,r3 = getMidPointFilteredArray(b,filterArray), getMidPointFilteredArray(g,filterArray), getMidPointFilteredArray(r,filterArray)

    filteredImage3 = mergeImage(b3,g3,r3)

    midPointFilteredImages.append(filteredImage3)
    
for i in range(len(images)):
    [imageName,extension] = imagesNames[i].split(".")
    cv.imwrite(imageName+"_median."+extension, medianFilteredImages[i])
    cv.imwrite(imageName+"_mean."+extension, meanFilteredImages[i])
    cv.imwrite(imageName+"_midpoint."+extension, midPointFilteredImages[i])
    

