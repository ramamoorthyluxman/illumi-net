import os
import shutil
from PIL import Image

def duplicate_folder_and_crop_images(folder_path, x, y, width, height):
    # Create the new folder name
    folder_name = os.path.basename(folder_path)
    new_folder_name = folder_name + "_sub_images"
    new_folder_path = os.path.join(os.path.dirname(folder_path), new_folder_name)

    # Create the new folder
    shutil.copytree(folder_path, new_folder_path)

    # Process images in the original folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            original_image_path = os.path.join(folder_path, filename)
            new_image_path = os.path.join(new_folder_path, filename)

            # Open the image
            with Image.open(original_image_path) as img:
                # Crop the image
                cropped_img = img.crop((x, y, x + width, y + height))
                
                # Save the cropped image in the new folder
                cropped_img.save(new_image_path)

    print(f"Folder duplicated and images cropped. New folder: {new_folder_path}")

# Example usage
folder_path = r"/work/imvia/ra7916lu/illumi-net/data/subset/readingPNG"  
x, y, width, height = 2, 0 , 608, 512  # Replace with your desired crop values

duplicate_folder_and_crop_images(folder_path, x, y, width, height)

