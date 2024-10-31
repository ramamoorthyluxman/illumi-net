import cv2
import numpy as np
import os

def crop_and_save_image(image_path, x, y, w, h):
    """
    Read an image, crop it to specified dimensions and save in the same directory
    
    Parameters:
    image_path (str): Path to the input image
    x (int): X coordinate of top-left corner of crop
    y (int): Y coordinate of top-left corner of crop
    w (int): Width of crop
    h (int): Height of crop
    
    Returns:
    str: Path to the saved cropped image
    """
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Crop the image
        cropped_image = image[y:y+h, x:x+w]
        
        # Generate output path
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{name}_cropped{ext}")
        
        # Save the cropped image
        cv2.imwrite(output_path, cropped_image)
        return output_path
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

def apply_mask_to_image(image_path, mask_path):
    """
    Apply a binary mask to an image
    
    Parameters:
    image_path (str): Path to the input image
    mask_path (str): Path to the binary mask image
    
    Returns:
    str: Path to the saved masked image
    """
    try:
        # Read the image and mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise ValueError("Could not read image or mask")
            
        # Ensure mask and image have same dimensions
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
        # Apply mask
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Generate output path
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{name}_masked{ext}")
        
        # Save the masked image
        cv2.imwrite(output_path, masked_image)
        return output_path
        
    except Exception as e:
        print(f"Error applying mask: {str(e)}")
        return None
    
import cv2
import numpy as np
import os

def convert_normal_map_to_npy(png_path):
    """
    Convert a normal map PNG image to NPY format
    
    Parameters:
    png_path (str): Path to the normal map PNG file
    
    Returns:
    str: Path to the saved NPY file
    """
    try:
        # Read the normal map image
        normal_map = cv2.imread(png_path, cv2.IMREAD_COLOR)
        if normal_map is None:
            raise ValueError(f"Could not read normal map at {png_path}")
            
        # Convert BGR to RGB
        normal_map = cv2.cvtColor(normal_map, cv2.COLOR_BGR2RGB)
        
        # Normalize the values from [0, 255] to [-1, 1]
        # Formula: (pixel_value / 127.5) - 1
        normal_map = (normal_map.astype(np.float32) / 127.5) - 1.0
        
        # Generate output path
        directory = os.path.dirname(png_path)
        filename = os.path.basename(png_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(directory, f"{name}_normal_map.npy")
        
        # Save as NPY file
        np.save(output_path, normal_map)
        
        # Verify the saved file
        test_load = np.load(output_path)
        if test_load.shape != normal_map.shape:
            raise ValueError("Verification of saved file failed")
            
        return output_path
        
    except Exception as e:
        print(f"Error converting normal map: {str(e)}")
        return None
    
# crop_and_save_image(r'/work/imvia/ra7916lu/illumi-net/data/subset/buddhaPNG/Normal_gt.png',2,0,608,512)
# apply_mask_to_image(r'/work/imvia/ra7916lu/illumi-net/data/subset/buddhaPNG_sub_images/albedo.png', '/work/imvia/ra7916lu/illumi-net/data/subset/buddhaPNG_sub_images/mask.png')

convert_normal_map_to_npy(r'/work/imvia/ra7916lu/illumi-net/data/subset/buddhaPNG/normal_map.png')