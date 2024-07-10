from PIL import Image
import numpy as np
import os
from skimage import color

# Function to split RGB image into R, G, B channels and save them as separate images
def split_rgb_channels(input_image_path, output_folder):
    if not os.path.isfile(input_image_path):
        raise FileNotFoundError(f"Input image file not found: {input_image_path}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the RGB image
    img = Image.open(input_image_path)
    img = img.convert('RGB')  # Ensure the image is in RGB mode

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Extract the R, G, B channels and create images
    red_channel = img_array.copy()
    red_channel[:, :, 1] = 0  # Set Green channel to 0
    red_channel[:, :, 2] = 0  # Set Blue channel to 0
    red_image = Image.fromarray(red_channel, 'RGB')
    
    green_channel = img_array.copy()
    green_channel[:, :, 0] = 0  # Set Red channel to 0
    green_channel[:, :, 2] = 0  # Set Blue channel to 0
    green_image = Image.fromarray(green_channel, 'RGB')
    
    blue_channel = img_array.copy()
    blue_channel[:, :, 0] = 0  # Set Red channel to 0
    blue_channel[:, :, 1] = 0  # Set Green channel to 0
    blue_image = Image.fromarray(blue_channel, 'RGB')

    # Save the images
    red_image.save(f'{output_folder}/red_channel.png')
    green_image.save(f'{output_folder}/green_channel.png')
    blue_image.save(f'{output_folder}/blue_channel.png')

    print(f"Saved Red channel as '{output_folder}/red_channel.png'")
    print(f"Saved Green channel as '{output_folder}/green_channel.png'")
    print(f"Saved Blue channel as '{output_folder}/blue_channel.png'")

    
def split_lab_channels(input_image_path, output_folder):
    """Splits RGB image into Y, Cb, Cr channels and saves them as separate color images."""
    if not os.path.isfile(input_image_path):
        raise FileNotFoundError(f"Input image file not found: {input_image_path}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load the RGB image
    img = Image.open(input_image_path)
    img = img.convert('RGB')  # Ensure the image is in RGB mode

      # Convert the image to a NumPy array
    img_array = np.array(img)

    # Convert RGB image to Lab color space using scikit-image
    lab_array = color.rgb2lab(img_array)

    # Extract the L, a, b channels
    L_channel = lab_array[:, :, 0]
    a_channel = lab_array[:, :, 1]
    b_channel = lab_array[:, :, 2]

    # Create Lab arrays for each channel visualization
    L_vis = np.zeros_like(lab_array)
    L_vis[:, :, 0] = L_channel
    L_vis_img = color.lab2rgb(L_vis)
    
    a_vis = np.zeros_like(lab_array)
    a_vis[:, :, 0] = 50  # Set L to mid-gray
    a_vis[:, :, 1] = a_channel
    a_vis_img = color.lab2rgb(a_vis)
    
    b_vis = np.zeros_like(lab_array)
    b_vis[:, :, 0] = 50  # Set L to mid-gray
    b_vis[:, :, 2] = b_channel
    b_vis_img = color.lab2rgb(b_vis)

    # Convert arrays back to PIL Images
    L_image = Image.fromarray((L_vis_img * 255).astype(np.uint8))
    a_image = Image.fromarray((a_vis_img * 255).astype(np.uint8))
    b_image = Image.fromarray((b_vis_img * 255).astype(np.uint8))

    # Save the images
    L_image.save(os.path.join(output_folder, 'L_channel_colored.png'))
    a_image.save(os.path.join(output_folder, 'a_channel_colored.png'))
    b_image.save(os.path.join(output_folder, 'b_channel_colored.png'))

    print(f"Saved L channel as '{output_folder}/L_channel_colored.png'")
    print(f"Saved a channel as '{output_folder}/a_channel_colored.png'")
    print(f"Saved b channel as '{output_folder}/b_channel_colored.png'")


# Example usage
input_image_path = './tree.jpg'  # Replace with the path to your RGB image
output_folder = './decomposed/'  # Replace with the path to your desired output folder
split_rgb_channels(input_image_path, output_folder)
split_lab_channels(input_image_path, output_folder)