import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import statistics
from statistics import mode

def getBBlist(imgpath):
    pre,ext = os.path.splitext(imgpath)
    txtpath = pre +'.txt'
    coors = []
    with open(txtpath, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        del lines[-1]       #last row is a '\n', so delete it
        for i,line in enumerate(lines):
            line = line.split(',')
            coor = [x_min,y_min,x_max,y_maxd] = list(map(int,line[0:4]))
            coors.append(coor)
        f.close()
    return coors

def cropROI(imgpath,des_path,margin=0):
    image = cv2.imread(imgpath)
    coors = getBBlist(imgpath)
    for i,[x_min,y_min,x_max,y_max] in enumerate(coors): 
        filename = os.path.join(des_path,str(i)+'.jpg')
        cv2.imwrite(filename,image[y_min-margin:y_max+margin,x_min-margin:x_max+margin])

def kmean(image,k=2,max_H = 0):
    #https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python#:~:text=Advertise-,How%20to%20Use%20K%2DMeans%20Clustering%20for%20Image%20Segmentation%20using,easier%20and%20more%20meaningful%20image.
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values) # convert to float
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)   # define stopping criteria
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)   # number of clusters (k)
    centers = np.uint8(centers) # convert back to 8 bit values
    print('centers',centers)
    labels = labels.flatten()       # flatten the labels array
    segmented_image = centers[labels.flatten()] # convert all pixels to the color of the centroids
    segmented_image = segmented_image.reshape(image.shape)  # reshape back to the original image dimension
    for x in range(len(centers)):
        # print('maxH = ' , max_H)
        if centers[x][2]> max_H:
            lower_hsv = centers[x]
            max_H = centers[x][2]
        else:
            pass
    # print('hsv = ',int(lower_hsv[0]),int(lower_hsv[1]),int(lower_hsv[2]))
    kmean_mask = cv2.inRange(segmented_image,(int(lower_hsv[0]),int(lower_hsv[1]),int(lower_hsv[2])),(255,255,255))
    masked_segmented = cv2.bitwise_and(image_hsv,image_hsv,mask = kmean_mask)
    return masked_segmented

def createLines(image,num_line=5):
    width = image.shape[1]
    height = image.shape[0]
    standard = min(width,height)
    interval = int(standard/num_line)
    x = [i for i in range(0,width+1,interval)]
    y = [j for j in range(0,height+1,interval)]
    
    hor_lines = [[(0,y[j]),(width,y[j])] for j in range(0,len(y))]
    ver_lines = [[(x[i],0),(x[i],height)] for i in range(0,len(x))] 

    return hor_lines,ver_lines
    """
    hor_lines = [[(0,y1),(width,y1)],[(0,y2),(width,y2)]]
    ver_lines = [[(x1,0),(x1,height)],[(x2,0),(x2,height)]]
    """

def findFirstWhite(image,hor_lines,ver_lines):
    """
    hor_lines = [[(0,y1),(width,y1)],[(0,y2),(width,y2)]]
    ver_lines = [[(x1,0),(x1,height)],[(x2,0),(x2,height)]]
    """
    leftWhites=[]
    rightWhites=[]
    topWhites=[]
    bottomWhites=[]
    width = image.shape[1]
    height = image.shape[0]
    for j,ver_line in enumerate(ver_lines):
        start = ver_line[0]     # start = (x1,0)
        end = ver_line[1]       # end = (x1,height)
        if j is not len(ver_lines)-1:
            x = start[0]  
        else:
            x= start[0]-1

        for y in range(0,height):
            if image[y][x] == 255:
                topWhites.append(y)
                break
       
        for y in range(height-1,-1 ,-1):
            print('bottom = ',image[y][x])
            if image[y][x] == 255:
                bottomWhites.append(y)
                break
    for i,hor_line in enumerate(hor_lines):
        start = hor_line[0]     # start = (0,y1)
        end = hor_line[1]       # end = (width,y1)
        if i is not len(hor_lines)-1:
            y = start[1]  
        else:
            y= start[1]-1
        for x in range(0,width):
            if image[y][x] == 255:
                leftWhites.append(x)
                break
        for x in range(width-1,-1,-1):
            if image[y][x] == 255:
                rightWhites.append(x)
                break
    
    return leftWhites,rightWhites,topWhites,bottomWhites

def determineBB(image,leftWhites,rightWhites,topWhites,bottomWhites,divider = 4):
    height = image.shape[0]
    width = image.shape[1]
    leftDict = {i:i for i in range(len(leftWhites))}
    rightDict = {i:i for i in range(len(rightWhites))}
    topDict = {i:i for i in range(len(topWhites))}
    bottomDict = {i:i for i in range(len(bottomWhites))}
    if len(topWhites) is not 0:
        average = np.sum(np.array(topWhites))/len(topWhites)
        for i,topWhite in enumerate(topWhites):
            if topWhites[i]>average:
                del topDict[i]
            elif topWhites[i]>int(height/divider):
                del topDict[i]
        # top = max(topWhites)
        ftopWhites = [topWhites[i] for i in topDict]
        top = max(ftopWhites)
    if len(bottomWhites) is not 0:
        average = np.sum(np.array(bottomWhites))/len(bottomWhites)
        for i,bottomWhite in enumerate(bottomWhites):
            if bottomWhites[i]<average:
                del bottomDict[i]
            elif int(height-bottomWhites[i])>int(height/divider):
                del bottomDict[i]
        fbottomWhites = [bottomWhites[i] for i in bottomDict]
        bottom = min(fbottomWhites)
    if len(leftWhites) is not 0:
        average = np.sum(np.array(leftWhites))/len(leftWhites)
        for i,leftWhite in enumerate(leftWhites):
            if leftWhites[i]>average:
                del leftDict[i]
            elif leftWhites[i]>int(width/divider):
                del leftDict[i]
        fleftWhites = [leftWhites[i] for i in leftDict]
        left = max(fleftWhites)
    if len(rightWhites) is not 0:
        average = np.sum(np.array(rightWhites))/len(rightWhites)
        for i,rightWhite in enumerate(rightWhites):
            if rightWhites[i]<average:
                del rightDict[i]
            elif int(width-rightWhites[i])>int(width/divider):
                del rightDict[i]
        frightWhites = [rightWhites[i] for i in rightDict]
        right = min(frightWhites)
    
    if len(topWhites) is 0:
        top = 0
    if len(bottomWhites) is 0:
        bottom = height
    if len(leftWhites) is 0:
        left = 0
    if len(rightWhites) is 0:
        right = width
    print('new bounding box coordinate = ',left,top,right,bottom) # equivalent to (x_min,y_min,x_max,y_max)
    return left,top,right,bottom

def drawBoxAndLine(image,hor_lines,ver_lines,x_min,y_min,x_max,y_max):
    print("image.shape = ",image.shape)
    cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (0,0,255), 1 )
    # # draw lines
    # for i,hor_line in enumerate(hor_lines):
    #     start = hor_line[0]
    #     end = hor_line[1]
    #     image = cv2.line(image, (start[0],start[1]), (end[0],end[1]), (0, 255, 0), 1)
    # for i,ver_line in enumerate(ver_lines):
    #     start = ver_line[0]
    #     end = ver_line[1]
    #     image = cv2.line(image, (start[0],start[1]), (end[0],end[1]), (0, 255, 0), 1)
    image = cv2.circle(image,(x_min,y_min),1,(255,0,0),-1)
    image = cv2.circle(image,(x_max,y_max),1,(255,0,0),-1)
    image = cv2.circle(image,(x_max,y_min),1,(255,0,0),-1)
    image = cv2.circle(image,(x_min,y_max),1,(255,0,0),-1)

# img_path = r'C:\Users\xiao-nan.gan\internProject\padTraining\images\myTest\binarized\55.jpg'
# filename = '55.jpg'
# image = cv2.imread(img_path)
# _, bw_image = cv2. threshold(image,127,255,cv2.THRESH_BINARY)
# print('shape = ',bw_image.shape)
# a,b,c = cv2.split(bw_image)
# hor_lines,ver_lines = createLines(image,num_line=10)
# leftWhites,rightWhites,topWhites,bottomWhites = findFirstWhite(a,hor_lines,ver_lines)
# print('left: ',leftWhites)
# print('right: ',rightWhites)
# print('top: ',topWhites)
# print('bottom: ',bottomWhites)
# x_min,y_min,x_max,y_max = determineBB(image,leftWhites,rightWhites,topWhites,bottomWhites)
# drawBoxAndLine(image,hor_lines,ver_lines,x_min,y_min,x_max,y_max)
# cv2.imwrite(r'C:\Users\xiao-nan.gan\internProject\padTraining\images\myTest\small.jpg',image)
# cv2.imshow(filename,image)
# key = cv2.waitKey(0)
# if key == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()

##################################### Cropped #############################################
imgpath = r'C:\Users\xiao-nan.gan\internProject\padTraining\images\myTest\ori\49043Bottom_6_3_6.jpg'
des_folder_path = r'C:\Users\xiao-nan.gan\internProject\padTraining\images\myTest\cropped'
cropROI(imgpath,des_folder_path)

##################################### Binarised ##############################################
img_folder_path = r'C:\Users\xiao-nan.gan\internProject\padTraining\images\myTest\cropped'
des_folder_path = r'C:\Users\xiao-nan.gan\internProject\padTraining\images\myTest\binarized'
for filename in os.listdir(img_folder_path):
    if filename.endswith('.jpg'):
        img_path = os.path.join(img_folder_path,filename)
        image = cv2.imread(img_path)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masked_segmented = kmean(image_hsv,k=2,max_H = 0)
        h,s,v = cv2.split(masked_segmented)
        threshed = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        output_path = os.path.join(des_folder_path,filename)
        cv2.imwrite(output_path,threshed[1])

##################################### Output ###################################################

# img_folder_path = r'C:\Users\xiao-nan.gan\internProject\padTraining\images\myTest\binarized'
# des_folder_path = r'C:\Users\xiao-nan.gan\internProject\padTraining\images\myTest\output'
# for filename in os.listdir(img_folder_path):
#     if filename.endswith('.jpg'):
#         img_path = os.path.join(img_folder_path,filename)
#         image = cv2.imread(img_path)
#         _, bw_image = cv2. threshold(image,127,255,cv2.THRESH_BINARY)
#         print('shape = ',bw_image.shape)
#         a,b,c = cv2.split(bw_image)
#         hor_lines,ver_lines = createLines(image,num_line=10)
#         leftWhites,rightWhites,topWhites,bottomWhites = findFirstWhite(a,hor_lines,ver_lines)
#         print('left: ',leftWhites)
#         print('right: ',rightWhites)
#         print('top: ',topWhites)
#         print('bottom: ',bottomWhites)
#         x_min,y_min,x_max,y_max = determineBB(image,leftWhites,rightWhites,topWhites,bottomWhites)
#         drawBoxAndLine(image,hor_lines,ver_lines,x_min,y_min,x_max,y_max)
#         output_path = os.path.join(des_folder_path,filename)
#         cv2.imwrite(output_path,image)
