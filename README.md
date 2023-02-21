# SBE 404B - Computer Vision

## CV_Final_Project

**Team 3**

**Submitted to: Dr. Ahmed Badwy and Eng. Laila Abbas**

Submitted by:

|              Name              | Section | B.N. |
|:------------------------------:|:-------:|:----:|
|   Esraa Mohamed Saeed   |    1    |   10  |
|   Alaa Tarek Samir   |    1    |  12  |
| Amira Gamal Mohamed  |    1    |  15  |
|   Fatma Hussein Wageh   |    2    |  8  |
| Mariam Mohamed Osama |    2    |  26  |

**The programming langusage is Python 3.8.8**

- **Libraries Used:**
  - Time
  - sys
  - os
  - math
  - glob
  - numpy==1.19.5
  - matplotlib==3.4.3
  - matplotlib-inline==0.1.3
  - opencv-contrib-python==4.5.3.56
  - opencv-python==4.5.3.56
  - PyQt5==5.15.6
  - PyQt5-Qt5==5.15.2
  - sklearn==0.0

## **How to run:**
**After running the code(main.py file) mainwindow will be opened on Face Detection tab**
- browse an image 
- click on detect and rectangles will be drawn around detected faces 

**For Face Recognition**
- browse an image 
- click on recognize and the result will be showed as the best matched image with your input 


## **Code Architecture:** <br>

## 1. Face Detection 
We used The **detectMultiScale()** function that detects objects. Since we are calling it on the face cascade, that’s what it detects.
This function takes 4 arguments 
1.	**grayscale** image.
2.	**scaleFactor**, since some faces may be closer to the camera, they would appear bigger than the faces in the back. The scale factor compensates for this.
3.	The detection algorithm uses a moving window to detect objects. **minNeighbors** defines how many objects are detected near the current one before it declares the face found. minSize, meanwhile, gives the size of each window.
## 2. Face Recognition
1. Reshape the whole images to 1D vectors.
2. Construct data matrix in shape of ( No. of pixels * No. of images ).
```python
training_images = np.ndarray(shape=(len(images_paths), height*width), dtype=np.float64)
#read the images and get the 1D vector
for i in range(len(images_paths)):
    path= training_path+'/'+ images_paths[i]
    read_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(read_image, (width, height))
#     print(resized_image.shape)
    training_images[i,:] = np.array(resized_image, dtype='float64').flatten()
```
3. Get the Mean Image.
4. Subtract the mean image from All images.
```python
##Get Mean Face##
    mean_face = np.zeros((1,height*width))

    for i in training_images:
        mean_face = np.add(mean_face,i)#to sum all images

    mean_face = np.divide(mean_face,float(len(images_paths))).flatten()# to get the mean face by dividing the summation by the length of images
    ##Normailze Faces##
    normalised_training = np.ndarray(shape=(len(images_paths), height*width))

    for i in range(len(images_paths)):
        normalised_training[i] = np.subtract(training_images[i],mean_face)#to substract mean face from each image

```
5. Get the Covariance matrix.
```python
#get the covariance matrix
def cov_mat(normalised_training):
    cov_matrix = ((normalised_training).dot(normalised_training.T))#dot product
    cov_matrix = np.divide(cov_matrix ,float(len(normalised_training)))
    return cov_matrix
```
6. Normalize the eigen vectors (each eigen vector is column of same size of
pixels number)
```python
#to get eigen faces and eigen vectors
def eigen_val_vec(cov_matrix):
    eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)
    eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]
    # Sort the eigen pairs in descending order:
    eig_pairs.sort(reverse=True)
    eigvalues_sort  = [eig_pairs[index][0] for index in range(len(eigenvalues))]
    eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]
    eigenfaces = preprocessing.normalize(eigvectors_sort)#normalize eigen vectors
```
7. Keep all vectors summing up eigen values to 90% and remove the rest.
```python
#to get the eigen faces till 90%
def get_reduced(eigenfaces,eigvalues_sort):
    var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)
    # Show cumulative proportion of varaince with respect to components

    #eigen faces components
    reduced_data=[]
    for i in (var_comp_sum):
        if i < 0.91:
            reduced_data.append(i)
    # print(reduced_data)
    # print(len(reduced_data))
    reduced_data = np.array(eigenfaces[:len(reduced_data)]).transpose()
```
8. Then map all images to new components and it will be in shape (column
vector of remaining eigen vectors length).
```python 
#data projection
def projected_data(training_images,reduced_data):
    proj_data = np.dot(training_images.transpose(),reduced_data)
    proj_data = proj_data.transpose()
    print(proj_data.shape)
    return proj_data

#get weights of images
def weights(proj_data,normalised_training):
    w = np.array([np.dot(proj_data,i) for i in normalised_training])
    return w
```    
9. To get best match image to test image. 

```python 
    w=weights(proj_data,normalised_training)
    w_unknown = np.dot(proj_data, normalised_uface_vector)#get the weight of test face
    euclidean_distance = np.linalg.norm(w - w_unknown, axis=1)#get the euclidean distance
    best_match = np.argmin(euclidean_distance)#get the index of the best matched one   
```

## 3. Preformance evaluation

In this part we measure the results performance in terms of accuracy and plotting the ROC curve. 
For even more accurate results than the eigen vectors method, we employed a RandomForest machine learning classifier

1. A function that recieves the data directory and returns the labels.

```python 
def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(filename)
            file_paths.append(filepath)
    labels = []
    for image_path in file_paths:
        label=(image_path.split(".")[0]).split("_")[1]
        labels.append(int(label))
    labels = np.array(labels)
    class_num=len(np.unique(labels))

    print('num of classes is:', class_num)
    return np.array(file_paths), labels  
```
2. Preparing the training and testing data. 

```python 
    def data_prep(direc, paths):
        images = np.ndarray(shape=(len(paths), height*width), dtype=np.float64)
        test_list=[]
        for i in range(len(paths)):
            path= direc+'/'+ paths[i]
            read_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            resized_image = cv2.resize(read_image, (width, height))
            images[i,:] = np.array(resized_image, dtype='float64').flatten()
            test_list.append(resized_image)
        print(images.shape)

        print(len(test_list))
        return (images)  
      
```
3. Accuracy evaluating function that recieves the actual and predicted labels:

```python 
    def accuracy(predictions, test_labels):
        l = len(test_labels)
        acc = sum([predictions[i]==test_labels[i] for i in range(l)])/l
        print('The testing accuracy is: ' + str(acc*100) + '%')
        return acc 
```

4. Creating the model , training and testing it.
```python 
    model = RandomForestClassifier()
    model.fit(training_images, labels)
    prob_vector = model.predict_proba(test_images)
    prediction = model.predict(test_images)  
```
5. ROC functions

This functions calculates the confucion matrix (TP, TN, FP, FN), the true positive rate and false pegative rates (TPR, FPR) for only one subject of all subjects. It recieves the model's predictions, the true labels, threshold and the subject that we are evaluating.

```python 
    def roc(probabilities, y_test, thresholds, subjectt):
        roc = np.array([])
        for threshold in thresholds:
            threshold_vector = np.greater_equal(probabilities, threshold).astype(int)
            results = np.where(y_test == 1)[0]
            tp, fp, tn, fn = 0,0,0,0
            for i in range(len(threshold_vector)):
                if i in results:
                    #which means that the actual value at these indices is 1
                    if threshold_vector[i] == 1:
                        tp +=1
                    else:
                        fn +=1
                else:
                    if threshold_vector[i] == 0:
                        tn +=1
                    elif threshold_vector[i] == 1:
                        fp +=1
    #         print(tp, fp, tn, fn)
            tpr = tp / (tp + fn)
            fpr = fp / (fp + tn)
    #         print(tpr, fpr)
    #         print('-'*60)
            roc = np.append(roc, [fpr, tpr])
        roc = roc.reshape(-1, 2)
        print(roc)
        return(roc)
    roc(prob_vector[:,20], y[:,20], thresholds, subject) 
```
6. Plotting the ROC: 

In these lines, we are considering subject 5 for testing the ROC function with the thresholds as shown 

```python 
    thresholds = np.arange(0.05, 1.05, 0.05)
    ROC = roc(prob_vector[:,4], y[:,4], threshlds, 4)
    plt.plot(ROC[:,0],ROC[:,1],color='#0F9D58')
    plt.title('ROC Curve',fontsize=20)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.show()
```
