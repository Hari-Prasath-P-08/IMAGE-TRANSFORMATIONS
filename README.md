# IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import numpy module as np and pandas as pd.

### Step2:
Assign the values to variables in the program.

### Step3:
Get the values from the user appropriately.

### Step4:
Continue the program by implementing the codes of required topics.

### Step5:
Thus the program is executed in google colab.

## Program:
```python
Developed By: Hari Prasath. P
Register Number: 212223230070

i)Image Translation

import numpy as np
import cv2
import matplotlib.pyplot as plt
# Read the input image
input_image = cv2.imread("Ajith Kumar Jump.jpg")
# Convert from BGR to RGB so we can plot using matplotlib
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
# Disable x & y axis
plt.axis('off')
# Show the image
plt.imshow(input_image)
plt.show()
# Get the image shape
rows, cols, dim = input_image.shape
# Transformation matrix for translation
M = np.float32([[1, 0, 50], [0, 1, 50],
[0, 0, 1]])
# Apply a perspective transformation to the image
translated_image = cv2.warpPerspective(input_image, M, (cols, rows))
# Disable x & y axis
plt.axis('off')
# Show the resulting image
plt.imshow(translated_image)
plt.show()

ii) Image Scaling

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'Ajith Kumar Jump.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis

# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)

iii)Image shearing



iv)Image Reflection




v)Image Rotation




vi)Image Cropping





```
## Output:
### i)Image Translation
![Screenshot 2024-03-22 144423](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/24369b95-11eb-4d89-9e88-7904bedcd9db)

![Screenshot 2024-03-22 144437](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/d2e311f2-94d3-4dba-b937-e011dca89be8)

### ii) Image Scaling

![Screenshot 2024-03-22 144423](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/5eccb2dc-6b60-4618-80ad-b4bf6e214f8e)

![Screenshot 2024-03-22 144656](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/cb7b4942-f503-4134-9fcf-aad7f27b2d5c)

### iii)Image shearing
<br>
<br>
<br>
<br>


### iv)Image Reflection
<br>
<br>
<br>
<br>



### v)Image Rotation
<br>
<br>
<br>
<br>



### vi)Image Cropping
<br>
<br>
<br>
<br>




## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
