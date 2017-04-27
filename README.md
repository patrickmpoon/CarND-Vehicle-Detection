
## Udacity Self Driving Car Nanodegree
## Submission for Project 5: Vehicle Detection and Tracking
### by Patrick Poon


---

## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

The `README.md` file at https://github.com/patrickmpoon/CarND-Vehicle-Detection serves as my write-up for this project.


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

#### Data Exploration

The training images used to create my classifier were obtained from Udacity per the instructions for this project, and were divided into the following:  [vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip).  An exploratory analysis shows that the images are relatively evenly divided by class:  vehicles (8792) and non-vehicles (8968).  Each image has a shape of (64, 64, 3) with data type `float32`.  Here is an example of each class:

<img src="./output_images/sample-car-notcar.png" />

I noticed that many of the images in the training set appeared to have been captured sequentially, so I made use of Python's `random.shuffle()` function whenever grabbing a subset for classification training.

#### HOG Testing

I experimented with different color spaces and `skimage.hog()` parameters, like `orientations`, `pixels_per_cell`, and `cells_per_block`.  I tested various images from both vehicle and non-vehicle classes and visualized them to see what the `skimage.hog()` output looks like.

Here is a visualization of both vehicle and non-vehicle HOG features, using the `YCrCb` color space for each color channel with HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


<img src="./output_images/ycbcr-hog-viz.png" />


The code for this step is contained in the third code cell of the **`worksheet.ipynb`** IPython notebook.  

#### 2. Explain how you settled on your final choice of HOG parameters.

The main parameters I tested to maximize the effectiveness of my classifier were `color_space`, `orient`, `pix_per_cell`, `cell_per_block`.  

To test color_space,  I tested RGB, HSV, HLS, YCrCb with the sample `color_space`, `orient`, `pix_per_cell`, `cell_per_block` values from "Module 40: Tips and Tricks for the Project" in the Udacity course materials for this project:
```
orient = 9
pix_per_cell = 8
cell_per_block = 2
```

Here are the results of my testing (All times are measured in seconds):

| Color Space | Testing Accuracy of SVC | Time to extract HOG features | Time to train SVC  | Time to predict 10 labels with SVC |
|:------:|:------:|:-----:|:-----:|:-------:|
| RGB   | 0.9483 |  12.27 | 1.34 | 0.00143 |
| HSV   | 0.9850 |  11.84 |  0.93 | 0.00133 |
| HLS   | 0.9850 |  10.63 | 0.91 | 0.00144 |
| **YCrCb** | **0.9900** | **10.57** | **1.06** | **0.00136** |

**YCrCb** was the clear winner here.

Next, I performed each successive tests for the `orient`, `pix_per_cell`, `cell_per_block` parameters, based on the parameter value that resulted in the highest accuracy.  Please refer to the cells under the **`TEST: [orient | pix_per_cell | cell_per_block]`** sections of the `worksheet.ipynb` file.  In cases where the testing accuracy results for different value parameters were the same, I chose the winning parameter value by considering time to extract HOG features, time to train the SVC and time to predict 10 labels with SVC.  The final winnning parameter selections were:

```
color_space = YCrCb
orient = 9
pix_per_cell = 6
cell_per_block = 2
```

*Note:*  `cell_per_block` had a higher score with a value of 1, but in order to maintain a shape that my classifier could work with, I had to set it to `2`.  `hog_channel = "ALL"` was chosen for a similar reason.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM and scaled the features to zero mean and unit variance by instantiating the **`StandardScaler`**, provided by `sklearn.preprocessing` library, then feeding its **`fit()`** function with the features as input in line 345 of the **`classifier.py`** file.  I took a random sampling of 1500 images from the training set to get the following training results:

```
Using: 9 orientations 6 pixels per cell and 2 cells per block
Feature vector length: 9564
9.24 Seconds to train SVC...
Test Accuracy of SVC =  0.9865
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I restricted my window search to image rows from 400 to 656, as it was unlikely that any cars would be found above the horizon at row 400, at least not until Uber launches its flying taxi service (please see [Fortune.com: Uber Picks Dallas and Dubai for Its Planned 2020 Flying Taxi Launch](http://fortune.com/2017/04/26/uber-dallas-dubai-2020-flying-taxi-launch/)).  I used a standard window size of 64x64 with an overlap of 0.5 (50%).

My initial tests only used one scale of 1.0, but this was less than optimal as cars closer to the horizon than those close to the capture vehicle were not detected well.  I tried a few different scales and settled on doing three sweeps with scales sizes and range specified in the table below in the next question.


<img src="./output_images/sliding_window_size.png" />

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I took a few actions to optimize the performance of my classifier.  First and foremost, as I mentioned above, I noticed that many of the training images were taken of the same vehicle sequentially.  To ensure that my classifier could generalize effectively, I used Python's random.shuffle() on the training set before feeding it to the `StandardScaler().fit()` function (Please see lines 298-299 and line 345 in the `classifier.py` file.

Next, I implemented a sliding window search only in the bottom half of the video file image where cars were more likely to be found.  I performed sweeps with at three different scales in different row ranges, as features would be detected more effectively the farther the vehicles were from our capture vehicle.  Here are the row ranges and scales I used (Please refer to lines 182-186 in the `pipeline.py` file):

| ystart | ystop | scale |
|:------:|:-----:|:-----:|
| 400    | 528   |  1.0  |
| 400    | 656   |  1.5  |
| 464    | 656   |  1.75 |


The following illustrates my search, using three different scale values and YCrCb 3-channel HOG features with spatially binned color and histograms of color in the feature vector:

<img src="./output_images/sliding_window_samples.png" />

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/me0Q7XJTWAY)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

All lines referenced in this section are in the `pipeline.py` file.

I tracked the positions of vehicle detections in each frame of video, by creating a Cars class (lines 12-21), which, when instantiated (line 225), would keep a list of the collective bounding boxes from the previous 20 frames.  From the detected bounding boxes I created a heatmap and then thresholded that map to identify vehicle positions (lines 208-212).  I then used `scipy.ndimage.measurements.label()` to identify clustered bounding boxes in the heatmap.  I then stipulated that each cluster signaled a vehicle.  I drew bounding boxes around each cluster detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

#### Five sequential video frames and their respective heatmaps:

<img src="./output_images/bounding_boxes_and_heatmaps.png" />

#### Output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all five frames:

<img src="./output_images/heatmap_across_frames.png" />


#### Final bounding boxes drawn onto the last frame in the group:

<img src="./output_images/resulting_bounding_boxes.png" />


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My initial testing and attempts resulted in many false positives appearing, as well as very jittery bounding boxes.  It was not until I figured out what the `writeup_template.md` was hinting at in question 2 of the Video Implementation section, that I figured out that keeping track of the previous frames and simply adding each bounding box to the `add_heat` function was key to smoothing out the bounding box tracking and minimizing the false positives.

It is hard to say where my pipeline is likely to fail.  The project video is pretty well lit with sunshine, so bad weather conditions might make it difficult for my classifier to detect vehicles effectively.  Most of the non-vehicles images used to train my classifier were of freeway images, so my pipeline might fail in city street conditions.

Training a convolutional neural network to process the training set images might make my pipeline more robust.  I did not augment my training data, but transforming the data images with different contrast and skew settings might also make it more robust.


