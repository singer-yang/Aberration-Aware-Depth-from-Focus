import cv2 as cv
import numpy as np
import re

def read_pfm(file):
    """
    Read an image in PFM format
    """
    with open(file, "rb") as f:
        header = f.readline().rstrip()
        color = True if header == b'PF' else False
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', f.readline().decode('ascii'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")
        scale = float(f.readline().rstrip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

def save_pfm_image(image_data, path):
    """
    Save PFM format image data to a specified path
    """
    # Check and replace NaN or infinite values
    image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
    # Normalize to 0-255 and convert to 8-bit integer
    normalized_image = cv.normalize(image_data, None, 0, 255, cv.NORM_MINMAX)
    image_8bit = np.uint8(normalized_image)
    # Save image
    cv.imwrite(path, image_8bit)
    print(f"Image saved to {path}")

# Path to the PFM image
image_path = r'D:\chenxireunion\be an engineer\Aberration-Aware-Depth-from-Focus-main\Aberration-Aware-Depth-from-Focus-main1\Aberration-Aware-Depth-from-Focus-main\dataset\Middlebury2014\Adirondack-perfect\disp0.pfm'
# Read PFM image
image_data = read_pfm(image_path)

# Exclude infinite values
finite_data = image_data[np.isfinite(image_data)]
# Calculate the maximum value excluding infinities
max_value_finite = np.max(finite_data)
print('Maximum depth value (excluding infinities):', max_value_finite)

# Calculate the minimum value
min_value = np.min(image_data)
print('Minimum depth value:', min_value)

# Save path
save_path = r'D:\chenxireunion\be an engineer\Aberration-Aware-Depth-from-Focus-main1\Aberration-Aware-Depth-from-Focus-main\results\output_image.png'
# Save PFM image
save_pfm_image(image_data, save_path)

# Print image shape and data type
print('Image Shape:', image_data.shape)
print('Data Type:', image_data.dtype)
