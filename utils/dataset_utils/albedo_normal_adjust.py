import cv2
import numpy as np
import os

def process_lp_and_apply_mask(folder_path, mask_path, output_folder=None):
    """
    Detect .lp file in the given folder, read it, and apply the given mask to images.
    
    Parameters:
    folder_path (str): Path to the folder containing .lp file and images
    mask_path (str): Path to the mask image
    output_folder (str): Optional path to save masked images. If None, creates 'masked_images' in folder_path
    
    Returns:
    tuple: (processed_files_count, light_positions)
        processed_files_count (int): Number of images processed
        light_positions (dict): Dictionary mapping image names to their light positions
    """
    try:
        # Find .lp file
        lp_files = [f for f in os.listdir(folder_path) if f.endswith('.lp')]
        if not lp_files:
            raise FileNotFoundError("No .lp file found in the specified folder")
        if len(lp_files) > 1:
            print("Warning: Multiple .lp files found. Using the first one.")
        
        lp_file = os.path.join(folder_path, lp_files[0])
        
        # Read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask at {mask_path}")
        
        # Normalize mask to range [0, 1]
        mask = mask.astype(float) / 255.0
        
        # Create output folder if not specified
        if output_folder is None:
            output_folder = os.path.join(folder_path)
        os.makedirs(output_folder, exist_ok=True)
        
        # Read .lp file and store light positions
        light_positions = {}
        processed_count = 0
        
        with open(lp_file, 'r') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Parse line
                parts = line.strip().split('\t')
                if len(parts) != 4:
                    print(f"Warning: Skipping invalid line: {line}")
                    continue
                    
                image_name = parts[0]
                light_pos = [float(x) for x in parts[1:]]
                light_positions[image_name] = light_pos
                
                # Construct image path
                image_path = os.path.join(folder_path, image_name)
                if not os.path.exists(image_path):
                    print(f"Warning: Image not found: {image_path}")
                    continue
                
                # Read image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Could not read image: {image_path}")
                    continue
                
                mask_resized = mask
                
                # Apply mask
                # Expand mask to 3 channels to match image
                mask_3channel = np.stack([mask_resized] * 3, axis=2)
                masked_image = (image * mask_3channel).astype(np.uint8)
                
                # Save masked image
                output_path = os.path.join(output_folder,image_name)
                cv2.imwrite(output_path, masked_image)
                
                processed_count += 1
                
                # Print progress every 10 images
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} images...")
        
        print(f"\nCompleted processing {processed_count} images")
        print(f"Masked images saved to: {output_folder}")
        
        return processed_count, light_positions
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return 0, {}


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
    

    
# crop_and_save_image(r'/work/imvia/ra7916lu/illumi-net/data/subset/readingPNG/Normal_gt.png',2,0,608,512)
# apply_mask_to_image(r'/work/imvia/ra7916lu/illumi-net/data/subset/readingPNG/albedo.png', '/work/imvia/ra7916lu/illumi-net/data/subset/readingPNG/mask.png')

# convert_normal_map_to_npy(r'/work/imvia/ra7916lu/illumi-net/data/subset/buddhaPNG/normal_map.png')

process_lp_and_apply_mask(r'/work/imvia/ra7916lu/illumi-net/data/subset/readingPNG', r'/work/imvia/ra7916lu/illumi-net/data/subset/readingPNG/mask.png', output_folder=None)
