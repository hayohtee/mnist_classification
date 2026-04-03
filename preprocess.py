import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_digit(digit: np.ndarray) -> None:
    image = np.reshape(digit, (28, 28))
    plt.imshow(image, cmap='gray')


def preprocess_image(image: np.ndarray) -> np.ndarray:
    # Binarize and Invert (MNIST is white digit on black background)
    # using THRESH_BINARY_INV for dark ink on light paper
    _, thresh = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Crop to the digit bounding box
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y + h, x:x + w]

    # Resize to 20x20 while preserving aspect ratio
    if w > h:
        new_w = 20
        new_h = int(h * (20 / w))
    else:
        new_h = 20
        new_w = int(w * (20 / h))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place in a 28x28 canvas and center by "Center of Mass"
    mnist_canvas = np.zeros((28, 28), dtype=np.uint8)
    pad_h, pad_w = (28 - new_h) // 2, (28 - new_w) // 2
    mnist_canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = digit_resized

    # Calculate Center of Mass (moments) and shift to the center (14, 14)
    M = cv2.moments(mnist_canvas)
    if M['m00'] != 0:
        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
        shift_x, shift_y = 14 - cx, 14 - cy
        M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        mnist_canvas = cv2.warpAffine(mnist_canvas, M_shift, (28, 28))

    # Normalization [0, 1] and flatten the image into 1D (1, 784)
    return (mnist_canvas.astype('float32') / 255.0).flatten().reshape(1, -1)
