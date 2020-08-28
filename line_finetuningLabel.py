import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import statistics
from statistics import mode
import shutil

def getBBlist(imgpath):
    pre,ext = os.path.splitext(imgpath)
    txtpath = pre +'.txt'
    _,filename = os.path.split(txtpath)
    coors = []
    labels = []
    with open(txtpath, 'r') as f:
        content = f.read()
        lines = content.split('\n')
        del lines[-1]       #last row is a '\n', so delete it
        for i,line in enumerate(lines):
            line = line.split(',')
            coor = [x_min,y_min,x_max,y_maxd] = list(map(int,line[0:4]))
            coor.append(int(line[-1]))
            coors.append(coor)
        f.close()
    return coors # coors = [[x_min,y_min,x_max,y_max,id0],[x_min,y_min,x_max,y_max,id1],...] 
                    

def tophat(image):
    # print(image.shape)
    width = image.shape[1]
    height = image.shape[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    kernel_size = max(width,height)//4
    if kernel_size <= 1:
        kernel_size = 2
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)
    inv = cv2.bitwise_not(thresh)
    image = cv2.bitwise_and(image,image,mask = inv)
    
    # print(image.shape,kernel_size)
    return image

def cropROI(imgpath,des_path,margin=0):
    print(imgpath)
    image = cv2.imread(imgpath)
    width = image.shape[1]
    height = image.shape[0]
    coors = getBBlist(imgpath)
    pre,post = os.path.split(imgpath)
    name,ext = os.path.splitext(post)

    for i,[x_min,y_min,x_max,y_max,id] in enumerate(coors): 
        filename = os.path.join(des_path,name+'_id_'+str(id)+ext)
        x_min_crop = x_min-margin 
        y_min_crop = y_min-margin
        x_max_crop = x_max+margin
        y_max_crop = y_max+margin

        x_min = x_min_crop if x_min_crop>=0 else x_min
        y_min = y_min_crop if y_min_crop>=0 else y_min
        x_max = x_max_crop if x_max_crop<=width else x_max
        y_max = y_max_crop if y_max_crop<=height else y_max
        small_image = image[y_min:y_max,x_min:x_max]

        cv2.imwrite(filename,small_image)

def kmean(image,k=2,max_H = 0):
    #https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python#:~:text=Advertise-,How%20to%20Use%20K%2DMeans%20Clustering%20for%20Image%20Segmentation%20using,easier%20and%20more%20meaningful%20image.
    image = cv2.medianBlur(image,5)
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values) # convert to float
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.5)   # define stopping criteria
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)   # number of clusters (k)
    centers = np.uint8(centers) # convert back to 8 bit values
    # print('centers',centers)
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
    if interval<= 0:
        interval=1
    x = [i for i in range(0,width+1,interval)]
    y = [j for j in range(0,height+1,interval)]
    
    hor_lines = [[(0,y[j]),(width,y[j])] for j in range(0,len(y))]
    ver_lines = [[(x[i],0),(x[i],height)] for i in range(0,len(x))] 

    return hor_lines,ver_lines
    """
    hor_lines = [[(0,y1),(width,y1)],[(0,y2),(width,y2)]]
    ver_lines = [[(x1,0),(x1,height)],[(x2,0),(x2,height)]]
    """

def findFirstWhite(image,hor_lines,ver_lines,direction='inward'):
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
    center = (center_x,center_y) = (int(width/2),int(height/2)) #center = (x,y)
    hor_dict = {}
    ver_dict = {}
    for j,ver_line in enumerate(ver_lines):
        start = ver_line[0]     # start = (x1,0)
        end = ver_line[1]       # end = (x1,height)
        if j is not len(ver_lines)-1:
            x = start[0]  
        else:
            x= start[0]-1

        if direction == 'inward':
            for y in range(0,height):
                if image[y][x] == 255:
                    topWhites.append(y)
                    break
        
            for y in range(height-1,-1 ,-1):
                if image[y][x] == 255:
                    bottomWhites.append(y)
                    break

        elif direction == 'outward':
            for y in range(center_y,-1,-1):
                if image[y][x] == 255:
                    topWhites.append(y)
                    ver_dict['top'+str(j)] = y
                    top_height = y
                    break
        
            for y in range(center_y,height,1):
                if image[y][x] == 255:
                    bottomWhites.append(y)
                    ver_dict['bottom'+str(j)] = y
                    bottom_height = y
                    break

    for i,hor_line in enumerate(hor_lines):
        start = hor_line[0]     # start = (0,y1)
        end = hor_line[1]       # end = (width,y1)
        if i is not len(hor_lines)-1:
            y = start[1]  
        else:
            y= start[1]-1

        if direction == 'inward':
            for x in range(0,width):
                if image[y][x] == 255:
                    leftWhites.append(x)
                    hor_dict['left'+str(i)] = x
                    break
            for x in range(width-1,-1,-1):
                if image[y][x] == 255:
                    rightWhites.append(x)
                    hor_dict['right'+str(i)] = x
                    break
        elif direction == 'outward':
            for x in range(center_x,-1,-1):
                if image[y][x] == 255:
                    leftWhites.append(x)
                    hor_dict['left'+str(i)] = x
                    break
            for x in range(center_x,width,1):
                if image[y][x] == 255:
                    rightWhites.append(x)
                    hor_dict['right'+str(i)] = x
                    break
    return leftWhites,rightWhites,topWhites,bottomWhites,ver_dict,hor_dict

def determineBB(image,leftWhites,rightWhites,topWhites,bottomWhites,divider = 2,direction='inward'):
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
        print(ftopWhites)
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

def conv_filter(image):
    sum_X = np.sum(image, axis=0)
    sum_Y = np.sum(image, axis=1)
    centerX, centerY = (image.shape[1]//2,image.shape[0]//2) #(x,y)

    conv_X = np.convolve(sum_X, [-1/5,2/5,1/5])
    conv_Y = np.convolve(sum_Y, [-1/5,2/5,1/5])

    ratio = 4.3/5
    threshold_X=np.median(conv_X) *ratio
    threshold_Y=np.median(conv_Y)*ratio

    outlier_Y,= np.where(conv_Y < threshold_Y)
    lower_list, = np.where(outlier_Y < centerY)
    lower_Y = outlier_Y [lower_list[-1]]
    upper_list, = np.where(outlier_Y > centerY)
    upper_Y = outlier_Y[upper_list[0]]

    # plt.plot(conv_Y, 'ro')
    # threshold_Y=np.median(conv_Y)*ratio
    # plt.plot(range(len(conv_Y)), [threshold_Y for i in range(len(conv_Y))])

    outlier_X,= np.where(conv_X < threshold_X)
    lower_list, = np.where(outlier_X < centerX)
    lower_X = outlier_X [lower_list[-1]]
    upper_list, = np.where(outlier_X > centerX)
    upper_X  = outlier_X[upper_list[0]]
    x_min,y_min,x_max,y_max = lower_X,lower_Y,upper_X,upper_Y
    return x_min,y_min,x_max,y_max

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

def verifyCircle(imgpath):
    image = cv2.imread(imgpath,0)
    image = cv2.medianBlur(image,5)
    height = image.shape[0]
    width = image.shape[1]
    if width>height:
        maxRadius = int(width/2+5)
        minRadius = int(width/3)
    else:
        maxRadius = int(height/2+5)
        minRadius = int(height/3)
    print('width,height = ',width,height)
    cimage = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)


    circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,20,
                                param1=50,param2=12,minRadius=minRadius,maxRadius=maxRadius)
    print(type(circles))

    try:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimage,(i[0],i[1]),i[2],(0,255,0),1)
            # draw the center of the circle
            cv2.circle(cimage,(i[0],i[1]),2,(0,0,255),-1)
            is_Circle=True
        return cimage,is_Circle
    except:
        print('no circle')
        is_Circle = False
        return image,is_Circle

    key = cv2.waitKey(0)
    if key==27:
        cv2.destroyAllWindows()

# img_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\_2crop\27.jpg'
# filename = '35.jpg'
# image = cv2.imread(img_path)
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# _, bw_image = cv2. threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# # print(bw_image.shape)
# # for row in bw_image:
# #     print(row)
# hor_lines,ver_lines = createLines(image,num_line=20)
# print('how many line',len(hor_lines),len(ver_lines))
# leftWhites,rightWhites,topWhites,bottomWhites,ver_dict,hor_dict = findFirstWhite(bw_image,hor_lines,ver_lines,direction='outward')
# print(ver_dict,hor_dict)
# print(len(leftWhites),len(rightWhites),len(topWhites),len(bottomWhites))
# print('left: ',leftWhites)
# print('right: ',rightWhites)
# print('top: ',topWhites)
# print('bottom: ',bottomWhites)
# x_min,y_min,x_max,y_max = determineBB(image,leftWhites,rightWhites,topWhites,bottomWhites,direction='outward')
# output = np.zeros_like(image)
# output[:,:,0] = bw_image
# output[:,:,1] = bw_image
# output[:,:,2] = bw_image
# drawBoxAndLine(output,hor_lines,ver_lines,x_min,y_min,x_max,y_max)
# cv2.imwrite(r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\4.jpg',output)
# cv2.imshow(r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\4.jpg',output)
# key = cv2.waitKey(0)
# if key == 27:         # wait for ESC key to exit
#     cv2.destroyAllWindows()

##################################### Cropped #############################################

# src_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\ori'
# des_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\_1crop'
# for filename in os.listdir(src_folder_path):
#     if filename.endswith('jpg'):
#         imgpath = os.path.join(src_folder_path,filename)
#         print('filename',imgpath)
#         cropROI(imgpath,des_folder_path)

##################################### Binarised ##############################################

# img_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\_1crop'
# des_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\binarized'
# for filename in os.listdir(img_folder_path):
#     if filename.endswith('.jpg'):
#         img_path = os.path.join(img_folder_path,filename)
#         image = cv2.imread(img_path)
#         image = tophat(image)
#         image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         masked_segmented = kmean(image_hsv,k=2,max_H = 0)
#         h,s,v = cv2.split(masked_segmented)
#         threshed = cv2.threshold(v,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#         path,filename = os.path.split(img_path)
#         output_path = os.path.join(des_folder_path,filename)
#         cv2.imwrite(output_path,threshed[1])
       
################################# Verify Circle################################################

# img_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\binarized'
# polygon_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\polygon'
# circle_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\circle'
# for filename in os.listdir(img_folder_path):
#     if filename.endswith('.jpg'):
#         img_path = os.path.join(img_folder_path,filename)
#         cimage,is_Circle = verifyCircle(img_path)
#         if is_Circle is True:
#             path,filename = os.path.split(img_path)
#             output_path = os.path.join(circle_folder_path,filename)
#             cv2.imwrite(output_path,cimage)
#         else:
#             path,filename = os.path.split(img_path)
#             output_path = os.path.join(polygon_folder_path,filename)
#             cv2.imwrite(output_path,cimage)

# #################################### _2 crop ###################################################

# img_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\polygon'
# des_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\_2crop_lineAlgo'
# for filename in os.listdir(img_folder_path):
#     if filename.endswith('.jpg'):
#         print('filename = ',filename)
#         img_path = os.path.join(img_folder_path,filename)
#         image = cv2.imread(img_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, bw_image = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#         hor_lines,ver_lines = createLines(image,num_line=20)
#         leftWhites,rightWhites,topWhites,bottomWhites,ver_dict,hor_dict = findFirstWhite(bw_image,hor_lines,ver_lines)
#         x_min,y_min,x_max,y_max = determineBB(image,leftWhites,rightWhites,topWhites,bottomWhites)
#         output = np.zeros_like(image)
#         output[:,:,0] = bw_image
#         output[:,:,1] = bw_image
#         output[:,:,2] = bw_image
#         drawBoxAndLine(image,hor_lines,ver_lines,x_min,y_min,x_max,y_max)
#         path,filename = os.path.split(img_path)
#         output_path = os.path.join(des_folder_path,filename)
#         cv2.imwrite(output_path,image)

##################################### _2 crop_no_label ###################################################

# img_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\polygon'
# des_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\_2crop_lineAlgo_no_label'
# for filename in os.listdir(img_folder_path):
#     if filename.endswith('.jpg'):
#         print('filename = ',filename)
#         img_path = os.path.join(img_folder_path,filename)
#         image = cv2.imread(img_path)
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         _, bw_image = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#         hor_lines,ver_lines = createLines(image,num_line=20)
#         leftWhites,rightWhites,topWhites,bottomWhites,ver_dict,hor_dict = findFirstWhite(bw_image,hor_lines,ver_lines)
#         x_min,y_min,x_max,y_max = determineBB(image,leftWhites,rightWhites,topWhites,bottomWhites)
#         output = image[y_min:y_max,x_min:x_max]
#         path,filename = os.path.split(img_path)
#         output_path = os.path.join(des_folder_path,filename)
#         cv2.imwrite(output_path,output)
      
################################### Convolution ################################################

img_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\_2crop_lineAlgo_no_label'
des_folder_path = r'C:\Users\xiao-nan.gan\internProject\finetuningLabel\myTest\_3crop_convAlgo_from_lineAlgo'
count = 0
for filename in os.listdir(img_folder_path):
    if filename.endswith('.jpg'):
        if count < 100000000:
            print('filename = ',filename)
            img_path = os.path.join(img_folder_path,filename)
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, bw_image = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
            x_min,y_min,x_max,y_max = conv_filter(bw_image)
            print(x_min,y_min,x_max,y_max)
            cv2.rectangle(image, (x_min,y_min), (x_max,y_max), (0,0,255), 1 )
            output_path = os.path.join(des_folder_path,filename)
            cv2.imwrite(output_path,image)
            count+= 1
