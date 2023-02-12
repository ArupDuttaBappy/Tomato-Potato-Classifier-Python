from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


# extract a 3D color histogram from the HSV color space
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # normalizing the histogram
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    # flattened histogram as the feature vector
    return hist.flatten()


def generate_features_and_labels(image_path):
    img_hist_feature = []
    img_class = []
    # loop over the input images
    for (i, imagePath) in enumerate(image_path):
        if i > 0 and i % 10 == 0:
            print("Processing [{}/{}]".format(i, len(image_path)))

        # load image, extract class label
        # path format: /path/to/dataset/{class}.{image_num}.jpg
        image = cv2.imread(imagePath)
        label = imagePath.split(os.path.sep)[-1].split(".")[0]

        # extract color histogram to characterize the color distribution of the image
        hist = extract_color_histogram(image)

        img_hist_feature.append(hist)
        img_class.append(label)

    return img_hist_feature, img_class


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--train", required=True, help="path to train dataset")
# ap.add_argument("-t", "--test", help="path to test dataset")
ap.add_argument("-k", "--neighbors", default=1, help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

# grab the list of training images
print("Loading training images...")
imagePaths_train = list(paths.list_images(args["train"]))
print("Extracting features and labels...")

train_img_hist_feature, train_img_class = generate_features_and_labels(imagePaths_train)

# train k-NN classifier
print("Training the knn model...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])
model.fit(train_img_hist_feature, train_img_class)

# grab the list of test images
# print("Loading test images...")
# imagePaths_test = list(paths.list_images(args["test"]))
# print("Extracting features and labels...")
#
# test_img_hist_feature, test_img_class = generate_features_and_labels(imagePaths_test)
#
# acc = model.score(test_img_hist_feature, test_img_class)
# print("histogram accuracy: {:.2f}%".format(acc * 100))

#########################################################
image = cv2.imread(os.path.basename('4.jpg'))
hist = extract_color_histogram(image)

prediction = model.predict(hist.reshape(1, -1))
print("Predicted result: ", prediction)
