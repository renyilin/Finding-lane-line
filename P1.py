
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# 
# ## Project: **Finding Lane Lines on the Road** 
# ***
# In this project, you will use the tools you learned about in the lesson to identify lane lines on the road.  You can develop your pipeline on a series of individual images, and later apply the result to a video stream (really just a series of images). Check out the video clip "raw-lines-example.mp4" (also contained in this repository) to see what the output should look like after using the helper functions below. 
# 
# Once you have a result that looks roughly like "raw-lines-example.mp4", you'll need to get creative and try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines.  You can see an example of the result you're going for in the video "P1_example.mp4".  Ultimately, you would like to draw just one line for the left side of the lane, and one for the right.
# 
# In addition to implementing code, there is a brief writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) that can be used to guide the writing process. Completing both the code in the Ipython notebook and the writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/322/view) for this project.
# 
# ---
# Let's have a look at our first image called 'test_images/solidWhiteRight.jpg'.  Run the 2 cells below (hit Shift-Enter or the "play" button above) to display the image.
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# ---

# **The tools you have are color selection, region of interest selection, grayscaling, Gaussian smoothing, Canny Edge Detection and Hough Tranform line detection.  You  are also free to explore and try other techniques that were not presented in the lesson.  Your goal is piece together a pipeline to detect the line segments in the image, then average/extrapolate them and draw them onto the image for display (as below).  Once you have a working pipeline, try it out on the video stream below.**
# 
# ---
# 
# <figure>
#  <img src="examples/line-segments-example.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above) after detecting line segments using the helper functions below </p> 
#  </figcaption>
# </figure>
#  <p></p> 
# <figure>
#  <img src="examples/laneLines_thirdPass.jpg" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your goal is to connect/average/extrapolate line segments to get output like this</p> 
#  </figcaption>
# </figure>

# **Run the cell below to import some packages.  If you get an `import error` for a package you've already installed, try changing your kernel (select the Kernel menu above --> Change Kernel).  Still have problems?  Try relaunching Jupyter Notebook from the terminal prompt.  Also, consult the forums for more troubleshooting tips.**  

# ## Import Packages

# In[1]:


#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression 
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read in an Image

# In[2]:


#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')


# ## Ideas for Lane Detection Pipeline

# **Some OpenCV functions (beyond those introduced in the lesson) that might be useful for this project are:**
# 
# `cv2.inRange()` for color selection  
# `cv2.fillPoly()` for regions selection  
# `cv2.line()` to draw lines on an image given endpoints  
# `cv2.addWeighted()` to coadd / overlay two images
# `cv2.cvtColor()` to grayscale or change color
# `cv2.imwrite()` to output images to file  
# `cv2.bitwise_and()` to apply a mask to an image
# 
# **Check out the OpenCV documentation to learn about these and discover even more awesome functionality!**

# ## Helper Functions

# Below are some helper functions to help get you started. They should look familiar from the lesson!

# In[3]:


import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def yellow_filter(img):
    """Applly yellow filter and then combine with canny image. 
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define yellow range.
    lower = np.array([0, 70, 70], dtype="uint8")
    upper = np.array([100, 255, 255], dtype="uint8")
    
    blur_img = gaussian_blur(img, kernel_size = 5)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(blur_img, blur_img, mask=mask)
    yellow_filter_output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    
    return yellow_filter_output

def combine_img(canny_img, yellow_filter_img):
    """Combine the yellow filter result and the canny result."""
    return cv2.bitwise_or(canny_img, yellow_filter_img)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def arg_mem(mem_list,*arg):
    """
    Record k,b during a period of time. If the memory list is full, the oldest arguments will 
    discard.
    
    Arguments:
        mem_list :  the list stores k,b.
        *arg: (k,b)
    """
    max_mem = 25
    mem_list.append(*arg)
    if len(mem_list) > max_mem :
        del(mem_list[0])
        
def cal_point(mem_list,width):
    """
    Calculate the endpoints of lane line.
    
    mem_list : the list stores k,b.
    width: the width of the image.
    return: p1(x1, y1), p2(x2, y2)
    """
    
    k_avg, b_avg = np.mean(np.array(mem_list),axis = 0)

    p1_y = width - 1
    p1_x = int((p1_y - b_avg)/k_avg)
    p2_y = int(340/540*width)
    p2_x = int((p2_y - b_avg)/k_avg)
    return (p1_x, p1_y), (p2_x, p2_y)

pt_left_mem = []
pt_right_mem = []

def draw_lines(img, gray_img, lines, color=[255, 0, 0], thickness=2, imshape= (540,960,3)):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    
#     for line in lines:
#         for x1,y1,x2,y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    width = img.shape[0]
    length = img.shape[1]
    pt_x_left = []
    pt_y_left = []
    pt_x_right = []
    pt_y_right = []   
    global pt_left_mem, pt_right_mem

    for line in lines:
        for x1,y1,x2,y2 in line:
            k = (y2-y1)/(x2-x1)
            if -0.8 < k <-0.4 and x1 < length/2 and x2 < length/2                 and y1 < width and y2 < width:
                # Record position of the points on line.
                    pt_x_left.append(x1)
                    pt_y_left.append(y1)    
                    pt_x_left.append(x2)
                    pt_y_left.append(y2)
            elif 0.8 > k > 0.4 and x1 > length/2 and x2 > length/2                     and y1 < width and y2 < width:              
                    pt_x_right.append(x1)
                    pt_x_right.append(x2)
                    pt_y_right.append(y1)
                    pt_y_right.append(y2)  
    
    # Create linear regression object
    regr = LinearRegression()
    
    if len(pt_x_left):
        # Train the model using the training sets
        regr.fit(np.reshape(pt_x_left,[-1,1]), np.reshape(pt_y_left,[-1,1]))
        k,b = regr.coef_[0], regr.intercept_
        arg_mem(pt_left_mem,(k,b))
    
    if len(pt_x_right):
        regr.fit(np.reshape(pt_x_right,[-1,1]), np.reshape(pt_y_right,[-1,1]))
        k,b = regr.coef_[0], regr.intercept_
        arg_mem(pt_right_mem,(k,b))
        
    p1_left, p2_left = cal_point(pt_left_mem, width)
    p1_right, p2_right = cal_point(pt_right_mem, width)
    
    cv2.line(img, p1_left, p2_left, color, thickness)
    cv2.line(img, p1_right, p2_right, color, thickness)

    # Show points on the images.
    show_points = 0
    if show_points:
        for i in range(len(pt_x_left)):
            cv2.circle(img, (int(pt_x_left[i]), int(pt_y_left[i])), 5, (0, 255, 0),-1) 
        for i in range(len(pt_x_right)):
            cv2.circle(img, (int(pt_x_right[i]), int(pt_y_right[i])), 5, (0, 255, 0),-1) 

        
def hough_lines(img, gray_img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, gray_img, lines, thickness=10, imshape = img.shape)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


# ## Test Images
# 
# Build your pipeline to work on the images in the directory "test_images"  
# **You should make sure your pipeline works well on these images before you try the videos.**

# In[4]:


import os
os.listdir("test_images/")


# ## Build a Lane Finding Pipeline
# 
# 

# Build the pipeline and run your solution on all test_images. Make copies into the `test_images_output` directory, and you can use the images in your writeup report.
# 
# Try tuning the various parameters, especially the low and high Canny thresholds as well as the Hough lines parameters.

# In[5]:


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.
import os 
test_path = "./test_images/"
im_list = os.listdir(test_path)


# Show the original images.
for i,img in enumerate(im_list):
    im_path = os.path.join(test_path,img)
    im_org = mpimg.imread(im_path)
    plt.subplot(321+i)
    plt.imshow(im_org)
    imshape = im_org.shape


# In[6]:


def detect_lane(img, vertices, low_threshold = 80, high_threshold = 150, theta = np.pi/180,
               rho = 2, threshold = 15, min_line_len = 20, max_line_gap = 10, kernel_size = 5):    
    
    gray_img = grayscale(img)
    blur_gray = gaussian_blur(gray_img, kernel_size)
    canny_img = canny(blur_gray, low_threshold, high_threshold)   
    yellow_filter_img = yellow_filter(img)
    cb_img = combine_img(canny_img, yellow_filter_img)
    masked_img = region_of_interest(cb_img, vertices)
    hough_img = hough_lines(masked_img, gray_img, rho, theta, threshold, min_line_len, max_line_gap)
    final_img = weighted_img(hough_img, img, α=0.7, β=1, γ=0.)
    
    return final_img


# In[7]:


# Create the image output path.
save_dir = './test_images_output'
if not os.path.isdir(save_dir):
     os.mkdir(save_dir)  


# In[8]:


# Define the vertices of detection region.
vertices = np.array([[(130, imshape[0]), (460, 320), (516, 320),
                      (895, imshape[0])]], dtype=np.int32)


# In[9]:


for i,img in enumerate(im_list):
    pt_left_mem = []
    pt_right_mem = []
    # Load image
    im_path = os.path.join(test_path,img)
    im_org = mpimg.imread(im_path)
    
    # Lane detection
    final_img = detect_lane(im_org, vertices)
    
    # Show image
    plt.subplot(321 + i)
    plt.imshow(final_img)
    
    # Save the final images to the test_images_output directory.    
    save_path = os.path.join(save_dir,img)
    cv2.imwrite(save_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 100])


# ## Test on Videos
# 
# You know what's cooler than drawing lanes over images? Drawing lanes over video!
# 
# We can test our solution on two provided videos:
# 
# `solidWhiteRight.mp4`
# 
# `solidYellowLeft.mp4`
# 
# **Note: if you get an import error when you run the next cell, try changing your kernel (select the Kernel menu above --> Change Kernel). Still have problems? Try relaunching Jupyter Notebook from the terminal prompt. Also, consult the forums for more troubleshooting tips.**
# 
# **If you get an error that looks like this:**
# ```
# NeedDownloadError: Need ffmpeg exe. 
# You can download it by calling: 
# imageio.plugins.ffmpeg.download()
# ```
# **Follow the instructions in the error message and check out [this forum post](https://discussions.udacity.com/t/project-error-of-test-on-videos/274082) for more troubleshooting tips across operating systems.**

# In[10]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[11]:


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)   
    imshape = image.shape
    vertices = np.array([[(130/960*imshape[1], imshape[0]), 
                          (450/960*imshape[1], 330/540*imshape[0]),
                          (530/960*imshape[1], 330/540*imshape[0]),
                          (895/960*imshape[1], imshape[0])]], dtype=np.int32)
    final_img = detect_lane(image, vertices)
    return final_img


# Let's try the one with the solid white lane on the right first ...

# In[12]:


pt_left_mem = []
pt_right_mem = []
n = 0
white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,4)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'white_clip.write_videofile(white_output, audio=False)')


# Play the video inline, or if you prefer find the video in your filesystem (should be in the same directory) and play it in your video player of choice.

# In[13]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))


# ## Improve the draw_lines() function
# 
# **At this point, if you were successful with making the pipeline and tuning parameters, you probably have the Hough line segments drawn onto the road, but what about identifying the full extent of the lane and marking it clearly as in the example video (P1_example.mp4)?  Think about defining a line to run the full length of the visible lane based on the line segments you identified with the Hough Transform. As mentioned previously, try to average and/or extrapolate the line segments you've detected to map out the full extent of the lane lines. You can see an example of the result you're going for in the video "P1_example.mp4".**
# 
# **Go back and modify your draw_lines function accordingly and try re-running your pipeline. The new output should draw a single, solid line over the left lane line and a single, solid line over the right lane line. The lines should start from the bottom of the image and extend out to the top of the region of interest.**

# Now for the one with the solid yellow lane on the left. This one's more tricky!

# In[14]:


pt_left_mem = []
pt_right_mem = []
n = 0
yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
get_ipython().run_line_magic('time', 'yellow_clip.write_videofile(yellow_output, audio=False)')


# In[15]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))


# ## Writeup and Submission
# 
# If you're satisfied with your video outputs, it's time to make the report writeup in a pdf or markdown file. Once you have this Ipython notebook ready along with the writeup, it's time to submit for review! Here is a [link](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md) to the writeup template file.
# 

# ## Optional Challenge
# 
# Try your lane finding pipeline on the video below.  Does it still work?  Can you figure out a way to make it more robust?  If you're up for the challenge, modify your pipeline so it works with this video and submit it along with the rest of your project!

# In[16]:


n = 0
pt_left_mem = []
pt_right_mem = []
challenge_output = 'test_videos_output/challenge.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
get_ipython().run_line_magic('time', 'challenge_clip.write_videofile(challenge_output, audio=False)')


# In[17]:


HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(challenge_output))

