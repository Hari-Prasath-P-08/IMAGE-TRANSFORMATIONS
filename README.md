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
Developed By: Hari Prasath. P
Register Number: 212223230070

### i)Image Translation:

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Read the input image
input_image = cv2.imread("AK Kabaddi Jump.jpg")
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
```

### ii) Image Scaling:

```python
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
image_url = 'AK Kabaddi Jump.jpg'  # Replace with your image URL or file path
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
```

### iii)Image shearing:

```python
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
image_url = 'AK Kabaddi Jump.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)
```

### iv)Image Reflection:

```python
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
image_url = 'AK Kabaddi Jump.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)

# Display original and reflected images
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)
```

### v)Image Rotation:

```python
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
image_url = 'AK Kabaddi Jump.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)
```

### vi)Image Cropping:

```python
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
image_url = 'Wall paper 2....jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)
```

## Output:
### i)Image Translation:

![Screenshot 2024-04-28 141339](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/60943b1d-2121-44b5-9c5f-506bd37baae0)

### ii) Image Scaling:

![Screenshot 2024-04-28 142107](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/fc3e02e8-53d1-4d1f-86d6-1d4fc70edd33)

### iii)Image shearing:

![Screenshot 2024-04-28 143811](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/82517454-e067-49ae-a2ff-71f07d05170d)

### iv)Image Reflection:

![Screenshot 2024-04-28 144005](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/ab28c03f-6d82-4e12-8b32-25d39598e9e4)
![Screenshot 2024-04-28 144016](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/6b4e21cf-12d2-40d3-bb12-32c279257572)

### v)Image Rotation:

![Screenshot 2024-04-28 144153](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/0c82e777-8c68-4bb9-830f-4af795e675a2)

### vi)Image Cropping:

![Screenshot 2024-04-28 145252](https://github.com/Hari-Prasath-P-08/IMAGE-TRANSFORMATIONS/assets/139455593/8d401067-0a08-40a0-8cd0-7a22ca1ac0eb)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
