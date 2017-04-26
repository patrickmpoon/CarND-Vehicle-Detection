import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# from lesson_functions import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from utilities import bin_spatial, color_hist, draw_boxes, get_training_data


TESTING = False


# Define a function to return HOG features and visualization
def get_hog_features(img,
                     orient,
                     pix_per_cell,
                     cell_per_block,
                     vis=False,
                     feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img,
                                  orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis,
                                  feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    else:
        features = hog(img,
                       orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis,
                       feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs,
                     color_space='RGB',
                     spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9,
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0,
                     spatial_feat=True,
                     hist_feat=True,
                     hog_feat=True):
    # Create a list to append feature vectors to
    features = []

    # Iterate through the list of images
    for file in imgs:
        file_features = []

        # Read in each one by one
        image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient,
                                                         pix_per_cell,
                                                         cell_per_block,
                                                         vis=False,
                                                         feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                                orient,
                                                pix_per_cell,
                                                cell_per_block,
                                                vis=False,
                                                feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img,
                 x_start_stop=[None, None],
                 y_start_stop=[None, None],
                 xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))

    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)

    # Initialize a list to append window positions to
    window_list = []

    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img,
                        color_space='RGB',
                        spatial_size=(32, 32),
                        hist_bins=32,
                        orient=9,
                        pix_per_cell=8,
                        cell_per_block=2,
                        hog_channel=0,
                        spatial_feat=True,
                        hist_feat=True,
                        hog_feat=True):

    # 1) Define an empty list to receive features
    img_features = []

    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)

    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)

    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)

    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient,
                                                     pix_per_cell,
                                                     cell_per_block,
                                                     vis=False,
                                                     feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel],
                                            orient,
                                            pix_per_cell,
                                            cell_per_block,
                                            vis=False,
                                            feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image and the list of windows to be searched (output of slide_windows())
def search_windows(img,
                   windows,
                   clf, scaler,
                   color_space='RGB',
                   spatial_size=(32, 32),
                   hist_bins=32,
                   hist_range=(0, 256),
                   orient=9,
                   pix_per_cell=8,
                   cell_per_block=2,
                   hog_channel=0,
                   spatial_feat=True,
                   hist_feat=True,
                   hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []

    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        # 4) Extract features for that window using single_img_features()
        features = single_img_features(test_img,
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)

        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        # 6) Predict using your classifier
        prediction = clf.predict(test_features)

        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)

    # 8) Return windows for positive detections
    return on_windows


if __name__ == "__main__":
    cars, notcars = get_training_data()
    shuffle(cars)
    shuffle(notcars)

    sample_size = 1500
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'HSV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 6  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 16  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [None, None]  # Min and max in y to search in slide_window()

    car_features = extract_features(cars,
                                    color_space=color_space,
                                    spatial_size=spatial_size,
                                    hist_bins=hist_bins,
                                    orient=orient,
                                    pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel,
                                    spatial_feat=spatial_feat,
                                    hist_feat=hist_feat,
                                    hog_feat=hog_feat)

    notcar_features = extract_features(notcars,
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient,
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)

    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()

    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t = time.time()

    image = mpimg.imread('./images/bbox-example-image.jpg')
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    # image = image.astype(np.float32)/255

    windows = slide_window(image,
                           x_start_stop=[None, None],
                           y_start_stop=y_start_stop,
                           xy_window=(96, 96),
                           xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image,
                                 windows,
                                 svc,
                                 X_scaler,
                                 color_space=color_space,
                                 spatial_size=spatial_size,
                                 hist_bins=hist_bins,
                                 orient=orient,
                                 pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel,
                                 spatial_feat=spatial_feat,
                                 hist_feat=hist_feat,
                                 hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    # plt.figure(figsize=(15, 15))
    # plt.imshow(window_img);

    # Save variables
    dist_pickle = {
        'svc': svc,
        'scaler': X_scaler,
        'orient': orient,
        'pix_per_cell': pix_per_cell,
        'cell_per_block': cell_per_block,
        'spatial_size': spatial_size,
        'hist_bins': hist_bins
    }

    fileObj = open('svc_pickle.p', 'wb')
    pickle.dump(dist_pickle, fileObj)
    fileObj.close()