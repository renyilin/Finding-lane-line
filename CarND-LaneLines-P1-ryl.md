# **Finding Lane Lines on the Road**

by RYL

---
### The goals of this project are the following:
* Make a pipeline that finds lane lines on the road.
* Reflect on the work in a written report.

[//]: # (Image References)

[image1]: pipeline.png
[image2]: ./test_images_output/solidWhiteRight_brokenline.jpg
[image3]: ./test_images_output/solidWhiteRight.jpg
[image4]: ./test_videos_output/challenge_GIF.gif

---

### Reflection

### 1. Description of the pipeline

The pipeline consists of 8 main steps which are summarized as the following. 

![Example given][image1]


1. Load an original image and convert it to the gray scaled image. (use `grayscale()`)
2. Process it with a Gaussian filter to blur and smooth the image. (use `gaussian_blur()`)
3. Apply the Canny algorithm to detect edges in the image. (use `canny()`)
4. Apply the yellow filter to detect the yellow region in the image. (use `yellow_filter()`)
5. Combine the yellow filter result and the canny result. (use `combine_img()`)
6. Mask edges out of the region of interest. (use `region_of_interest()`)
7. Apply the Hough transform algorithm to get segments of lane line. (use `hough_lines()`)
8. Process the segments of lines and draw lane lines in the image. (use `draw_lines()`)

Except yellow filter, all of the other steps have been well described in the course material. The yellow filter is added for detecting yellow region in the image. We can get a better result with it, especially when detecting yellow lane in the image or video, such as the scene in the `challenge.mp4`.

One of the final results of the lane finding pipeline is shown in the image as below.

![broken_line_result][image2]


### 2. Description of the `draw_lines()` function modification
In order to draw a single lane line on the left and right in the image, the `draw_lines()` function was modified.
The algorithm has three main steps which are described as below.

1. Extract effective points $(x_1, y_1), (x_2, y_2) $ from both ends of all edges by Hough algorithm; Decide them left or right lane points by computing the slope $(y_2-y_1)/(x_2-x_1)$.
- the effective segments/points by Hough algorithm should have reasonable slope and lay in reasonable region.
2. Apply linear regression to get the slope $k$ and the intercept $b$ for left and right lane line; record the $(k,b)$. 
- In this project, I use `sklearn.linear_model.LinearRegression` to fit the line.
3. Average $(k,b)$ in the last $n$ steps as the $(k,b)$ of the current lane line. Calculate the end point and the start point of the lane line.
- The use of averaged $(k,b)$ is to obtain robust lane lines in video streams.
- The value of $n$ is a tuning parameter.

The results of using modified `draw_lines()` function shows as following when applied to the above example. 

![draw_lines][image3]

The below video shows the result of the most challenging task (`challenge.mp4`).

![challenge_pic][image4]

The other image or video results can be found in [this html file](http://CarND-LaneLines-P1-jungpil.html).


### 3. Potential shortcomings and possible improvements suggestion
There are several potential shortcomings:
1. This method only fits lane lines by linear model. However, the lane is not always straight line. So using polynomial or other models to fit lane lines could help improve the accuracy.
2. All of the parameters are manually tuned, so that they may not be the optimal values for other images or videos under different situations. This problem could be solved by introducing an adaptive parameter algorithm.
3. This algorithm may fail when the cars drive in dark environment. In this case, most of the vision methods won't work anymore. So other assistant methods, such as HD map and localization, should be introduced. 
