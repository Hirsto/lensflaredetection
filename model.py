import sys

import matplotlib.image as mpimg

import numpy as np

from skimage import feature
from skimage import io


from sklearn.svm import LinearSVC






def process(img_nm):
    
    

    
    img = io.imread(img_nm, True)

 


    

    lbp = feature.local_binary_pattern(img, 24,
            12, method="uniform")

    (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, 24 + 3),
            range=(0, 12 + 2))

    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)



    return hist

def trainModel(tfeatures, tpred):
    
    lsvc = LinearSVC(tol=1e-5)
    
    lsvc.fit(tfeatures, tpred)
    


    #cv_results = cross_validate(lsvc, tfeatures, tpred, cv=8)

    return lsvc


    
    
    



features = np.load("./training_features.npy")
pred = np.load("training_predictions.npy")


model = trainModel(features, pred)

i = 1
while i < len(sys.argv):
    
    
    img_feat = process(sys.argv[i]).reshape(1,-1)
    print(np.around(model.predict(img_feat)[0]))
    i = i +1
    



