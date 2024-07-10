import cv2
import os

def convert_images_to_L_channel(input_folder, output_folder):
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Construct full file path
            img_path = os.path.join(input_folder, filename)
            
            # Read the image
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Failed to load image {filename}")
                continue
            
            # Convert the image to Lab color space
            lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            
            # Split the Lab image to L, a, b channels
            L_channel, a_channel, b_channel = cv2.split(lab_image)
            
            # Save the L channel image
            output_path = os.path.join(output_folder, f"L_{filename}")
            cv2.imwrite(output_path, L_channel)
            print(f"Saved L channel of {filename} to {output_path}")

# Example usage:
input_folder = './ground_truth/'
output_folder = './luminance_only/'
convert_images_to_L_channel(input_folder, output_folder)
