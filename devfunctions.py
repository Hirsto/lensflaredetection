import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import glob
import numpy as np
import cv2 as cv
from skimage import feature
from skimage import measure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC


features = []
pred = []


# def Classify():

#     features = np.


def connectedComponent(img):

    blurred = cv.GaussianBlur(img, (11, 11), 0)

    thresh = cv.threshold(blurred, 235, 255, cv.THRESH_BINARY)[1]
    
    
    thresh = cv.erode(thresh, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=4)


    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    mask2 = mask

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue

        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv.countNonZero(labelMask)

        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        if numPixels > 50:
            mask = cv.add(mask, labelMask)
    return mask


def findCircles(img, oimg, fname, tvalue, e1, e2, c1 ,c2, cdistance, mins, maxs):
    #find edges


    #img = cv.GaussianBlur(img, (11, 11), 0)
    gray = cv.cvtColor(oimg, cv.COLOR_BGR2GRAY)
    
    
    

    threshimg = cv.threshold(img, tvalue, 255, cv.THRESH_BINARY)[1]
    
    #thresh = img
    thresh = cv.erode(threshimg, None, iterations=2)
    thresh = cv.dilate(thresh, None, iterations=6)

    #thresh = cv.medianBlur(thresh, 15)

    edges = cv.Canny(thresh,e1,e2)

    plt.subplot(2,2,4),plt.imshow(thresh)
    plt.title('smoothed threshold image'), plt.xticks([]), plt.yticks([])

    

    oimg = np.bitwise_or(oimg, edges[:,:,np.newaxis])

    
    
    #cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

    #this function has internal cannay but couldn't get results so preprocessed edges
    
    circles = cv.HoughCircles(edges,cv.HOUGH_GRADIENT,1, cdistance, param1=c1,param2=c2,minRadius=mins,maxRadius=maxs)
    hist = np.zeros((1, 10))

    
    

    
    if circles is None:
        print("NA")
        plt.savefig( fname , dpi= 200)
        plt.clf()
        return circles, hist
    if type(circles[0]) == None:
        print("NA")
        plt.savefig( fname , dpi= 200)
        plt.clf()
        return circles, hist

    
    

    if not np.any(circles):
        print("NA")
        plt.savefig( fname , dpi= 200)
        plt.clf()
        return circles, hist
    #circles = np.uint8(circles)

    #filter circles to find what we're looking for

    

    
    brightCircles = []
    for i in circles[0]:
        
       
        if filterBrightCircles(i, gray):
            
            brightCircles.append(i)
    
    brightCircles = np.array(brightCircles)
    
    
   
    
    if len(brightCircles)>0:
        
        for i in brightCircles:
            lbp = getLBP(gray, i)
            

            hist = np.add(hist, lbp)
            #print(i)
            # draw the outer circle
            cv.circle(oimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(oimg,(i[0],i[1]),2,(0,0,255),3)
        
                
        
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
    print("breakpoint 4")
    plt.subplot(2,2,1),plt.imshow(edges)
    plt.title('Detection image'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,2),plt.imshow(oimg)
    plt.title('Circles Detected'), plt.xticks([]), plt.yticks([])

    plt.subplot(2,2,3),plt.imshow(img)
    plt.title('image with edges'), plt.xticks([]), plt.yticks([])

    plt.savefig( fname , dpi= 200)
    plt.clf()

    print("printed " +fname)
    

    return circles, hist




#requires grayscale
def filterBrightCircles(circle, img):
    
    
    if img[int(circle.item(1))][int(circle.item(0))] < 230:


        return False



    return isCircleGradient(circle, img, -10)

def filterFlares(circle, img):
    return


def getLBP(img, circle):



    shape = img.shape
    mask = np.zeros(shape, dtype=np.uint8)

    mask = cv.circle(mask, (circle[0],circle[1]), circle[2], 255, -1)


    final_img = cv.bitwise_or(img, img, mask=mask)



    desc = LocalBinaryPatterns(8,2)

    
    lbp, hist = desc.describe(img)

    # Display images, used for debugging
    #cv.imshow('Original Image', img)
    #cv2.imshow('Sketched Mask', image_mark)
    
    
    

    
    # plt.subplot(2,2,1), plt.hist(hist, bins='auto')
    # plt.title('hist'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,2),plt.imshow(final_img)
    # plt.title('Detection image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(2,2,3),plt.imshow(img)
    # plt.title('original'), plt.xticks([]), plt.yticks([])
    # plt.show()
    
    
    #cv.imshow('Output Image', final_img)
    #cv2.imshow('Table of Images', table_of_images)
    cv.waitKey(4) # Wait for a keyboard event 
    return hist






#returns true if there is more than one angle of the circle that returns
#a gradient above the threshold
def isCircleGradient(circle, img, threshold):
    
    #x and y could be swapped -> opencv circle output -> can't find it in docs
    xpos = int(circle.item(1))
    ypos = int(circle.item(0))
    radius = int(circle.item(2))
    isarc = False
    brightnessCenter = img[xpos][ypos]


    gradientcount = 0
    c = 0
    while c < 8:
        
        x = xpos
        y = ypos

        r = radius

        bness = [brightnessCenter]

        i = 0
        while i < 2:

            i = i +1
            r = int(r/2)

            if  (c % 2):
                r = int(r *0.7) #Sin45 since at 45 degree angle ->otherwise we're measuring a box not a circle


            if (c > 0) and (c < 4):
                x = x + r
            elif (c > 4) and (c < 8):
                x = x - r
                
            if (c > 2) and (c < 6):
                y = y - r
            elif (c != (2 or 6)):
                y = y + r

            if (((x and y) > 0) and ((x and y) < 600)):
                bness.append(img[y][x])

        if len(bness) > 2:

            
            loss1 = bness[0] - bness[1]
            loss2 = bness[1] - bness[2]

            avg = (loss1 + loss2)/2
            #print(avg)

            if avg > threshold:
                gradientcount = gradientcount +1
        c = c+1

    if gradientcount >= 0:
        isarc = True
            

    return isarc

    



class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")



        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        

        
        return lbp, hist
	    

def process(img_nm, note, num, isflare):
    img2 = mpimg.imread(img_nm)
    img = cv.imread(img_nm, 0)
    oimg = img
    img3 = img2

    img_feat = []



    #img3 = ReduceRGB(img3)
    
    plt.subplot(4,2,1),plt.imshow(img3,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])


    #img = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)

    
    # T_otsu = mahotas.thresholding.otsu(img)
    # seeds,_ = mahotas.label(img < T_otsu)
    # labeled = mahotas.cwatershed(img.max() - img, seeds)

    # plt.imshow(labeled), plt.show()


    #img = cv.imread(img_nm)
    

    #findCircles(img, img2)

    desc = LocalBinaryPatterns(24,12)

    #plt.hist(img.ravel(), bins=256)
    #plt.savefig(folder + str(num) + "p" )
    #plt.imsave(str(num) + note, ReduceRGB(img))
    
    #plt.imsave(str(num) + note, ReduceRGB(CircleDet(img)))

    #img = resizeimg(img, 40)


    plt.subplot(4,2,5),plt.imshow(img)
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    
   
 

    #img = cv.threshold(img,220,255,cv.THRESH_TOZERO)[1]
    
    lbp, hist = desc.describe(img)
    #print(hist)
    
    #img_feat.append(hist)

    pred.append(isflare)

    

    
    
    
    

    

    #plt.subplot(4,2,2), plt.hist(hist, bins='auto')
    #plt.title('LBP Hist'), plt.xticks([]), plt.yticks([])



    #sobely = cv.Sobel(lbp,cv.CV_64F,0,1,ksize=25)

    #hsv = cv.cvtColor(img3,cv.COLOR_BGR2HSV)
    #chist = cv.calcHist([img2],[0],None,[256],[0,256])

    #plt.subplot(4,2,3),plt.imshow(sobely,cmap = 'gray')
    #plt.title('sobely'), plt.xticks([]), plt.yticks([])

    #hist = desc.describe(sobely)
    #plt.subplot(4,2,4), plt.hist(chist, bins='auto')
    #plt.title('color hist'), plt.xticks([]), plt.yticks([])

    
    # plt.savefig( str(num) + note , dpi= 200)


    #--- First obtain the threshold using the greyscale image ---
    ret,th = cv.threshold(img,220,255, 0)

    #--- Find all the contours in the binary image ---
    _, contours,hierarchy = cv.findContours(th,2,1)
    cnt = contours
    big_contour = []

    max = 0
    for i in cnt:
        area = cv.contourArea(i) #--- find the contour having biggest area ---
        if(area > max):
            max = area - area*0.2
            big_contour.append(i)

    #final = cv.drawContours(img, big_contour, -1, (255,255,0), 6)
    #print(max)
    
    # plt.subplot(4,2,6),plt.imshow(final)
    # plt.title('Original' + str(max)), plt.xticks([]), plt.yticks([])
    

    

    #img_feats = np.add(hist, max)
    hl = []
    #hl.append(max)


    #circles, hist = findCircles(img, img2, str(num) + note + "-CD2", 240, 100, 200, 40, 10, 80, 10, 500)
    for i in hist:
        #print(i[0])
        hl.append(i)

    print(str(hl) + "hist")


    
    #img_feat.append(hl)
    #img = ReduceRGB(img)


    
    #findCircles(img, img2, str(num) + note + "-CD2", 100, 200, 200, 40)
    #findCircles(img, img2, str(num) + note + "-CD3", 200, 400, 200, 20)
    #findCircles(img, img2, str(num) + note + "-CD4", 50, 200, 80, 20)
    
    
    
    #hog(img)

   # img = findcontours(img)
    
    #ORB

    # #img = cv.imread('simple.jpg',0)
    # # Initiate ORB detector
    # orb = cv.ORB_create()
    # # # find the keypoints with ORB
    # kp = orb.detect(img,None)
    # # # compute the descriptors with ORB
    # kp, des = orb.compute(img, kp)
    # # # draw only keypoints location,not size and orientation
    # img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    # plt.imshow(img2), plt.show()

    #plt.hist(img.ravel(), bins=256)]

    #cvgt(img)
    plt.savefig( str(num) + note , dpi= 200)
    plt.clf()


    
    #findCircles(img, oimg, fname, tvalue, e1, e2, c1 ,c2, cdistance, mins, maxs)
    #findCircles(sobely, img2, str(num) + note + "-CD1", 240, 100, 200, 20, 5, 80, 10, 500)
    
    #findCircles(img, img2, str(num) + note + "-CDSpot", 240, 100, 200, 150, 10, 80, 8, 30)
    #plt.imsave(str(num) + note, img)


    features.append(hl)
    

def imageiter():
    flares = glob.glob('./flare/*.JPG')
    goods = glob.glob('./good/*.JPG')
    v =0
    while v < 38:
        v+=1
        print(v)



        process(flares[v], "-Flare", v, 1)




        process(goods[v], "-Good", v, 0)

def resizeimg(img, scale):
    scale_percent = scale # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    return cv.resize(img, dim, interpolation = cv.INTER_AREA)        

def testp(img, num):
    plt.hist(img.ravel(), bins=256)


    plt.savefig(str(num) + " fig")
    plt.imsave(str(num), img)
    plt.clf()


def ReduceRGB(z):

    #todo change algorithm relative to average brightness
    img = np.copy(z)
    v = 0
    for i in range(img.size):

        if i  % 3 == 0:
            avg = (img.item(i-1) + img.item(i) + img.item(i +1))/3

            if (avg > 260) or (avg < 230):
                np.put(img, v-1, 0)
                np.put(img, v, 0)
                np.put(img, v+1, 0)
                


        
        v +=1
    return img


def cvgt(img):
    

    laplacian = cv.Laplacian(img,cv.CV_64F)
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=1)
    plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    #plt.show()

def findcontours(im):
    
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 200, 255, 0)
    im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnt = contours
    cv.drawContours(im, cnt, 0, (0,255,0), 3)
    return im

def CircleDet(img):
    print("done1")
    img = cv.medianBlur(img,5)
    cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                                param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    return cimg

def process2(img_nm, note, num, isflare):


    
    img = cv.imread(img_nm, 0)


    #desc = LocalBinaryPatterns(24,12)
    #lbp, hist = desc.describe(img)
    circles, hist2 = findCircles(img, img2, str(num) + note + "-CD2", 240, 100, 200, 40, 10, 80, 10, 500)

    
    pred.append(isflare)
    
    plt.savefig( str(num) + note , dpi= 200)
    plt.clf()

    
 

    
    #findCircles(img, oimg, fname, tvalue, e1, e2, c1 ,c2, cdistance, mins, maxs)
    #findCircles(img, img2, str(num) + note + "-CD2", 240, 100, 200, 40, 10, 80, 10, 500)
    #findCircles(img, img2, str(num) + note + "-CDSpot", 240, 100, 200, 150, 10, 80, 8, 30)
    #plt.imsave(str(num) + note, img)
    print(hist)

    features.append(hist)




def classify(features, pred):
    train_features, test_features, train_labels, test_labels = train_test_split(features, pred, test_size = 0.3)

    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 100)
    rf = LinearSVC(tol=1e-5)
    
    

    print(train_features)
    print(train_labels)
    


    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    print(predictions)
    print(test_labels)
    predictions = np.around(predictions)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

   

    #print(confusion_matrix(test_labels,predictions))
    print(classification_report(test_labels,predictions))
    print(accuracy_score(test_labels,predictions))


    cv_results = cross_validate(rf, features, pred, cv=8)
    print(np.mean(cv_results['test_score']))
    
    



#imageiter()

features = np.load("features-3cleaned.npy")
pred = np.load("predictions-3cleaned.npy")



features = np.array(features)
pred = np.array(pred)

print("items")
print(features)
print(pred)


#np.save("features-3cleaned", features)
#np.save("predictions-3cleaned", pred)


classify(features, pred)

#cvgt()


# print(img)
# testp(img,1)


# img = mpimg.imread('./good/G0011262.JPG')
# #print(img)


# plt.hist(img.ravel(), bins=256)


# plt.savefig('good.png')
# plt.imsave('0', img)