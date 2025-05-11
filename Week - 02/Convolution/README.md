# Week 2: Understanding Convolution & CNNs

## Introduction to Signal Processing and Convolution

Welcome to Week 2 of our Seasons of Code project on intermediate missing frames in animated videos! Last week, we learned about Python basics and essential libraries. This week, we'll explore a fascinating concept from signal processing: **convolution**.

### Why is this important for our project?

Generating intermediate frames in videos requires understanding how images are processed and transformed. Convolution is a fundamental operation that helps us extract features from images, detect patterns, and ultimately understand the content - all critical steps for our frame interpolation project.

## Understanding Convolution

### Different Perspectives: The Frame of Reference Analogy

Imagine observing a ball falling from a moving airplane:
- From the ground, you see a parabolic path (due to gravity and forward momentum)
- From the plane, you see it moving straight down

The same object behaves differently depending on your frame of reference. Similarly, in signal processing, we can view functions from different "frames of reference" or **paradigms**:
- **Time domain**: How a signal changes over time
- **Frequency domain**: What frequencies make up the signal

### What is Convolution?

Convolution is a mathematical operation that expresses how the shape of one function is modified by another. It's represented by the symbol * (not to be confused with multiplication).

If we have an input signal x(t) and a system characterized by h(t) (called the impulse response or kernel), the output y(t) is:

y(t) = h(t) * x(t)

This is calculated as:

y(t) = ∫ h(t-τ)x(τ) dτ

In plain language: for each point in time, we multiply the input signal with a flipped and shifted version of our kernel, then sum up all the results.

```python
# Let's import the libraries we'll need
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2
from IPython.display import Image, display
from skimage import color, data, restoration, io
import warnings
warnings.filterwarnings('ignore')
```

### Visualizing Convolution in 1D

Let's start by visualizing what convolution looks like in one dimension. We'll convolve a simple sine wave with a rectangular pulse.

```python
# Create a time vector
t = np.linspace(-10, 10, 1000)

# Create a sine wave as our input signal x(t)
frequency = 1  # Hz
x = np.sin(2 * np.pi * frequency * t)

# Create a rectangular pulse as our kernel h(t)
h = np.zeros_like(t)
h[(t >= -1) & (t <= 1)] = 1
h = h / np.sum(h)  # Normalize the kernel

# Perform the convolution
y = np.convolve(x, h, mode='same')

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

ax1.plot(t, x)
ax1.set_title('Input Signal x(t)')
ax1.set_xlabel('Time')
ax1.grid(True)

ax2.plot(t, h)
ax2.set_title('Kernel h(t)')
ax2.set_xlabel('Time')
ax2.grid(True)

ax3.plot(t, y)
ax3.set_title('Output Signal y(t) = h(t) * x(t)')
ax3.set_xlabel('Time')
ax3.grid(True)

plt.tight_layout()
plt.show()
```

Notice how the convolution with the rectangular pulse has "smoothed out" our sine wave. This is because convolution with a rectangle is effectively computing a moving average.

### Understanding the Time and Frequency Domains

Now, let's explore the relationship between the time and frequency domains. We'll create a signal that's a mixture of two sine waves at different frequencies.

```python
# Create a time vector
t = np.linspace(0, 1, 1000)

# Create a signal with two frequency components
f1 = 5  # Hz
f2 = 20  # Hz
x = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Compute the Fourier transform to see the frequency components
X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), t[1] - t[0])

# Plot the signal in time domain
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.plot(t, x)
plt.title('Signal in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the magnitude spectrum (frequency domain)
plt.subplot(2, 1, 2)
plt.plot(freqs[:len(freqs)//2], np.abs(X[:len(X)//2]))
plt.title('Signal in Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.tight_layout()
plt.show()
```

In the frequency domain plot, you should see two clear peaks at 5 Hz and 20 Hz, corresponding to our two sine wave components.

### Why is the Frequency Domain Important?

The frequency domain gives us a different perspective on our signal. It shows us the "ingredients" that make up the signal, just like a recipe lists ingredients for a dish.

**Real-world example:** Consider human speech. In the time domain, we see a complex waveform that's hard to analyze. In the frequency domain, we can identify which frequencies are present and their strengths, which correspond to different phonemes (speech sounds).

**For image processing:** 
- Low frequencies represent smooth, gradual changes in an image (like background)
- High frequencies represent rapid changes, like edges and details
- Noise typically lives in the high-frequency range

### Convolution in the Frequency Domain

One of the most powerful properties of convolution is that it becomes multiplication in the frequency domain:

If y(t) = h(t) * x(t) in the time domain, then Y(f) = H(f) × X(f) in the frequency domain

This property makes many calculations much simpler!

Let's now move from 1D signals to 2D images, where convolution becomes even more powerful.

## Convolution in Image Processing

In images, convolution works the same way but in two dimensions. We slide a small matrix (called a kernel or filter) across the image, multiply the overlapping values, sum them up, and place the result in the output image.

### Edge Detection with Convolution

One of the most common applications of convolution in image processing is edge detection. Let's look at how we can use the Sobel operator to detect edges in an image.

```python
# Load an image
try:
    # Try to load the provided image
    image = cv2.imread('input_1.jpg')
    if image is None:  # If image couldn't be loaded
        # Use a sample image from scikit-image
        image = data.camera()
    else:
        # Convert BGR to RGB (OpenCV loads images in BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
except:
    # Use a sample image if there's any error
    image = data.camera()

# If the image is color, convert to grayscale
if len(image.shape) == 3 and image.shape[2] == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
else:
    gray_image = image

# Display the original image
plt.figure(figsize=(10, 10))
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()
```

### Understanding the Sobel Operator

The Sobel operator is a pair of 3×3 kernels that detect horizontal and vertical edges:

Horizontal Sobel kernel (detects vertical edges):
```
[-1 0 1]
[-2 0 2]
[-1 0 1]
```

Vertical Sobel kernel (detects horizontal edges):
```
[ 1  2  1]
[ 0  0  0]
[-1 -2 -1]
```

Let's apply these kernels to our image and see what happens:

```python
# Define the Sobel kernels
sobel_x = np.array([[-1, 0, 1], 
                     [-2, 0, 2], 
                     [-1, 0, 1]])

sobel_y = np.array([[1, 2, 1], 
                     [0, 0, 0], 
                     [-1, -2, -1]])

# Apply the kernels using convolution
edges_x = cv2.filter2D(gray_image, -1, sobel_x)
edges_y = cv2.filter2D(gray_image, -1, sobel_y)

# Combine the edges
edges = np.sqrt(edges_x**2 + edges_y**2)
edges = edges / edges.max() * 255  # Normalize to 0-255

# Display the results
plt.figure(figsize=(15, 15))

plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(np.abs(edges_x), cmap='gray')
plt.title('Horizontal Edges (Sobel X)')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(np.abs(edges_y), cmap='gray')
plt.title('Vertical Edges (Sobel Y)')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(edges, cmap='gray')
plt.title('Combined Edges')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### The Effect of Kernel Size on Edge Detection

The size of the kernel has a significant impact on the results of edge detection. Let's experiment with different kernel sizes:

```python
def create_sobel_kernel(size):
    """Create Sobel kernels of a given size."""
    # For sizes > 3, we'll create approximations by scaling up
    if size == 3:
        return sobel_x, sobel_y
    
    # Create larger kernels (simple approach - there are better methods)
    center = size // 2
    sobel_x_large = np.zeros((size, size))
    sobel_y_large = np.zeros((size, size))
    
    # Fill outer columns for x kernel
    for i in range(size):
        weight = 1.0
        if i < center:
            weight = (i + 1) / (center + 1)
        elif i > center:
            weight = (size - i) / (center + 1)
        
        sobel_x_large[i, 0] = -weight
        sobel_x_large[i, -1] = weight
    
    # Fill outer rows for y kernel
    for j in range(size):
        weight = 1.0
        if j < center:
            weight = (j + 1) / (center + 1)
        elif j > center:
            weight = (size - j) / (center + 1)
        
        sobel_y_large[0, j] = weight
        sobel_y_large[-1, j] = -weight
        
    return sobel_x_large, sobel_y_large

def detect_edges(image, kernel_size):
    """Detect edges in an image using Sobel operators of the specified size."""
    # Create Sobel kernels of the given size
    kernel_x, kernel_y = create_sobel_kernel(kernel_size)
    
    # Apply the kernels
    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)
    
    # Combine the edges
    edges = np.sqrt(edges_x**2 + edges_y**2)
    edges = edges / edges.max() * 255  # Normalize to 0-255
    
    return edges

# Apply edge detection with different kernel sizes
kernel_sizes = [3, 5, 9, 15]
results = []

for size in kernel_sizes:
    edges = detect_edges(gray_image, size)
    results.append(edges)

# Display the results
plt.figure(figsize=(20, 10))

for i, (size, edges) in enumerate(zip(kernel_sizes, results)):
    plt.subplot(1, len(kernel_sizes), i+1)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Kernel Size: {size}x{size}')
    plt.axis('off')

plt.tight_layout()
plt.show()
```

### Understanding Kernel Size Effects in Edge Detection

#### Larger Kernels
**Pros:**
- Capture broader, smoother transitions
- Can detect low-frequency, thick edges

**Cons:**
- More likely to blur fine details
- May introduce false positives in noisy regions (i.e., amplify noise as edges)

#### Smaller Kernels
**Pros:**
- Capture sharp, fine edges (high-frequency content)
- Better at preserving details

**Cons:**
- Miss subtle transitions (like smooth shading)
- Less robust against gradient blending

### Research Challenge:
Can you find an optimal kernel size for edge detection? Or perhaps a combination of multiple kernel sizes? 

**Hints:**
- You could use machine learning to learn the optimal kernel size
- You could combine results from multiple kernel sizes
- You could adapt the kernel size based on local image characteristics

## Convolution Neural Networks (CNNs)

Now that we understand convolution and how it can be used for tasks like edge detection, we can begin to see why it's so important in deep learning for image processing.

In a Convolutional Neural Network (CNN):
1. Kernels are not hand-designed (like Sobel) but learned from data
2. Multiple kernels are applied to extract different features
3. The outputs are processed through non-linear functions and pooling operations
4. This process is repeated in multiple layers

The first layers often learn to detect simple features like edges (similar to Sobel), while deeper layers combine these to detect more complex patterns like textures, shapes, and eventually entire objects.

For an excellent visual explanation of CNNs and how they relate to the convolution operation we've been discussing, watch the 3Blue1Brown video "But what is a convolution?" (https://www.youtube.com/watch?v=KuXjwB4LzSA).

## Part 2: Reverse Engineering Image Filters

Now for the fun part! Let's see if we can reverse-engineer an image filter by determining the kernel that was used to transform an original image into a filtered one.

### Creating a Known Filter

First, let's create a filtered version of our image using a known kernel. We'll use a Gaussian blur filter.

```python
# Create a Gaussian blur kernel
kernel_size = 15
sigma = 5
gaussian_kernel = cv2.getGaussianKernel(kernel_size, sigma)
gaussian_kernel = gaussian_kernel @ gaussian_kernel.T  # Convert to 2D

# Apply the kernel to our image
blurred_image = cv2.filter2D(gray_image, -1, gaussian_kernel)

# Display original and blurred images
plt.figure(figsize=(15, 7))

plt.subplot(1, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gaussian_kernel, cmap='viridis')
plt.title('Gaussian Kernel')
plt.colorbar()

plt.tight_layout()
plt.show()
```

### Determining the Unknown Kernel

Now, let's pretend we don't know what kernel was used. All we have is the original image and the filtered result. Can we reverse-engineer the kernel?

In the frequency domain, we can derive:

Y(f) = H(f) × X(f)

Therefore:

H(f) = Y(f) / X(f)

We can then convert back to the spatial domain to get our kernel h(t).

Let's implement this approach using the Wiener deconvolution method:

```python
def estimate_kernel(original, filtered, kernel_size=15):
    """
    Estimate the convolution kernel that transforms the original image into the filtered image.
    
    This is a simplified approach and works best with simple filters.
    """
    # Convert to float for better precision
    original = original.astype(float)
    filtered = filtered.astype(float)
    
    # Create a delta function (impulse) in the center of a zero image
    impulse = np.zeros_like(original)
    center_y, center_x = impulse.shape[0] // 2, impulse.shape[1] // 2
    impulse[center_y, center_x] = 1.0
    
    # Use Wiener deconvolution to estimate the PSF (Point Spread Function, aka kernel)
    psf = restoration.wiener(filtered, impulse, 0.1)
    
    # Extract the central portion of the PSF as our kernel
    half_size = kernel_size // 2
    kernel = psf[center_y-half_size:center_y+half_size+1, center_x-half_size:center_x+half_size+1]
    
    # Normalize the kernel
    kernel = kernel / np.sum(kernel)
    
    return kernel

# Estimate the kernel
estimated_kernel = estimate_kernel(gray_image, blurred_image)

# Apply the estimated kernel to the original image
reconstructed_image = cv2.filter2D(gray_image, -1, estimated_kernel)

# Display the results
plt.figure(figsize=(20, 10))

plt.subplot(2, 3, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Filtered Image (Target)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image (Using Estimated Kernel)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(gaussian_kernel, cmap='viridis')
plt.title('Original Kernel')
plt.colorbar()

plt.subplot(2, 3, 5)
plt.imshow(estimated_kernel, cmap='viridis')
plt.title('Estimated Kernel')
plt.colorbar()

plt.subplot(2, 3, 6)
error_image = np.abs(blurred_image - reconstructed_image)
plt.imshow(error_image, cmap='hot')
plt.title(f'Error (Mean: {np.mean(error_image):.2f})')
plt.colorbar()

plt.tight_layout()
plt.show()
```

## Analyzing the Results

Let's analyze the results of our kernel estimation:

```python
# Calculate metrics to evaluate the quality of our kernel estimation
mse = np.mean((blurred_image - reconstructed_image) ** 2)
max_error = np.max(np.abs(blurred_image - reconstructed_image))
kernel_similarity = np.sum(gaussian_kernel * estimated_kernel) / (np.sqrt(np.sum(gaussian_kernel**2)) * np.sqrt(np.sum(estimated_kernel**2)))

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Maximum Error: {max_error:.4f}")
print(f"Kernel Similarity (cosine): {kernel_similarity:.4f}")

# Let's also look at cross-sections of the kernels
plt.figure(figsize=(12, 5))

# Cross-section along the center row
center_row = kernel_size // 2
plt.subplot(1, 2, 1)
plt.plot(gaussian_kernel[center_row, :], 'b-', label='Original Kernel')
plt.plot(estimated_kernel[center_row, :], 'r--', label='Estimated Kernel')
plt.title('Kernel Cross-section (Horizontal)')
plt.legend()
plt.grid(True)

# Cross-section along the center column
plt.subplot(1, 2, 2)
plt.plot(gaussian_kernel[:, center_row], 'b-', label='Original Kernel')
plt.plot(estimated_kernel[:, center_row], 'r--', label='Estimated Kernel')
plt.title('Kernel Cross-section (Vertical)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## The Challenge of Kernel Estimation

Kernel estimation is an inverse problem - we try to recover the cause (the kernel) from the effect (the filtered image). This is generally more difficult than the forward problem (applying a kernel to get a filtered image).

Some challenges include:
- Noise in the images makes estimation harder
- Edge effects can impact the quality of estimation
- Multiple kernels might produce similar results
- Some information might be lost in the convolution process

## CNNs for Understanding Video Frames

As we move towards our goal of generating intermediate frames, convolutional neural networks (CNNs) will play a crucial role. Here's how:

1. **Feature Extraction**: CNNs excel at extracting hierarchical features from images, from low-level edges to high-level semantic content.

2. **Optical Flow Estimation**: A common approach to frame interpolation is to estimate optical flow between frames. CNNs can learn to predict how pixels move between frames.

3. **Frame Synthesis**: Advanced CNN architectures can learn to generate intermediate frames directly, especially when combined with techniques like adversarial training.

## Next Steps

In the coming weeks, we'll build on these fundamentals:

1. Explore neural network architectures specifically designed for video processing
2. Study optical flow techniques for tracking motion between frames
3. Implement and train models to generate intermediate frames
4. Evaluate different approaches and fine-tune our solution

## Conclusion

Convolution is a powerful mathematical operation that connects signal processing theory with modern deep learning approaches. By understanding how convolution works at a fundamental level:

- We've seen how simple operations like blurring and edge detection can be formulated as convolutions
- We've explored the relationship between the time/spatial domain and the frequency domain
- We've attempted to reverse-engineer filters by estimating kernels
- We've introduced how these concepts translate to convolutional neural networks

These concepts will form the foundation of our work in video frame interpolation. Next week, we'll dive deeper into neural network architectures specifically designed for this task!

## Homework Assignment

1. Experiment with different kernel sizes and shapes for edge detection on your own images
2. Try to reverse-engineer more complex filters (e.g., emboss, sharpen, etc.)
3. Read about optical flow estimation and how it relates to convolution
4. [Optional] Check out the paper "Deep Voxel Flow for Video Super Resolution" to see how CNN-based approaches handle frame interpolation

Feel free to share your results in our project chat, and don't hesitate to ask questions!
