import os
import io
import hashlib
from PIL import Image, ImageOps, ImageEnhance, UnidentifiedImageError
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Manager
import random

# Paths
path_with_masks = '/Users/markzhuang/Desktop/Project/with_mask'
path_without_masks = '/Users/markzhuang/Desktop/Project/without_mask'
output_path_with_masks = '/Users/markzhuang/Desktop/Project/with_mask_processed'
output_path_without_masks = '/Users/markzhuang/Desktop/Project/without_mask_processed'

# Constants
SIZE_SMALL = (32, 32)
SIZE_LARGE = (128, 128)
REQUIRED_IMAGES_PER_CLASS = 5000

# Augmentation parameters
ROTATION_DEGREES = 30  # ± this value
BRIGHTNESS_ADJUSTMENT = 0.1  # up to this value (10%)
FLIP_PROBABILITY = 0.5  # 50% chance

# Ensure the output directories exist
os.makedirs(output_path_with_masks, exist_ok=True)
os.makedirs(output_path_without_masks, exist_ok=True)

# Function to downscale images and generate MD5 hashes for deduplication
def downscale_and_hash(image_path):
    try:
        # Open the image and resize it to a smaller size for faster hashing
        with Image.open(image_path) as image:
            image_small = image.resize(SIZE_SMALL)
        
        # Convert the downscaled image to a byte array and calculate its MD5 hash
        image_bytes = np.array(image_small).tobytes()
        hash_md5 = hashlib.md5(image_bytes).hexdigest()
        
        # Return the original image path and the computed hash
        return (image_path, hash_md5)
    except UnidentifiedImageError:
        # If the image file is not identified, skip it and return None for the hash
        print(f"Skipping file: {image_path}, not a valid image.")
        return (image_path, None)

# Function to perform image augmentation and normalization
def augment_and_normalize(image_path):
    try:
        with Image.open(image_path) as img:
            # Ensure image is in RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_resized = img.resize(SIZE_LARGE)
            img_rotated = img_resized.rotate(random.uniform(-ROTATION_DEGREES, ROTATION_DEGREES))
            enhancer = ImageEnhance.Brightness(img_rotated)
            img_bright = enhancer.enhance(random.uniform(1 - BRIGHTNESS_ADJUSTMENT, 1 + BRIGHTNESS_ADJUSTMENT))
            if random.random() < FLIP_PROBABILITY:
                img_bright = ImageOps.mirror(img_bright)
            return img_bright
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to process and save augmented images
def process_and_save_augmented(args):
    image_path, output_directory, shared_unique_hashes, lock, class_prefix, class_counter = args
    img = augment_and_normalize(image_path)
    if img:
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
        hash_md5 = hashlib.md5(img_byte_array).hexdigest()
        lock.acquire()
        try:
            if hash_md5 not in shared_unique_hashes:
                unique_id = class_counter[class_prefix]
                class_counter[class_prefix] += 1
                filename = f"{class_prefix}_{unique_id:05d}.png"
                img.save(os.path.join(output_directory, filename))
                shared_unique_hashes[hash_md5] = None
        finally:
            lock.release()

def process_images(class_path, output_path, shared_unique_hashes, lock, class_prefix, class_counter):
    # List of image paths
        image_paths = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Create a pool of workers for downscaling and hashing
        with Pool() as pool:
            # Downscale and hash images
            hashes = list(tqdm(pool.imap_unordered(downscale_and_hash, image_paths), total=len(image_paths), desc="Hashing images"))

        # Remove duplicates and get unique hashes
        unique_hashes = set()
        unique_image_paths = []

        for image_path, hash in hashes:
            if hash is not None and hash not in unique_hashes:
                unique_hashes.add(hash)
                unique_image_paths.append(image_path)

        print(f"Unique images in '{class_prefix}' after hashing: {len(unique_hashes)}")
        print(f"Number of images to be augmented: {len(unique_image_paths)}")  # Ensure this prints the correct number

        # Augment images using multiprocessing, but only for unique images
        with Pool(processes=os.cpu_count()) as pool:
            args = [(path, output_path, shared_unique_hashes, lock, class_prefix, class_counter) for path in unique_image_paths]
            assert len(args) == len(unique_image_paths), "The length of args does not match the number of unique images"
            # Use len(unique_image_paths) for accurate progress bar count
            list(tqdm(pool.imap_unordered(process_and_save_augmented, args), total=len(unique_image_paths), desc="Augmenting images"))
            return len(unique_image_paths)

# Main function
def main():
    with Manager() as manager:
        shared_unique_hashes = manager.dict()  # Shared dictionary
        lock = manager.Lock()  # Shared lock
        class_counter = manager.dict({'with_mask': 0, 'without_mask': 0})  # Class counters

        # Process images with masks
        unique_with_mask = process_images(path_with_masks, output_path_with_masks, shared_unique_hashes, lock, 'with_mask', class_counter)
        print(f"Processed {unique_with_mask} unique 'with_mask' images.")

        # Process images without masks
        unique_without_mask = process_images(path_without_masks, output_path_without_masks, shared_unique_hashes, lock, 'without_mask', class_counter)
        print(f"Processed {unique_without_mask} unique 'without_mask' images.")

        # Check the number of images and perform additional augmentations if necessary
        for class_prefix, output_path in [('with_mask', output_path_with_masks), ('without_mask', output_path_without_masks)]:
            image_files = set(f for f in os.listdir(output_path) if f.endswith(('.png', '.jpg', '.jpeg')))  # Ensure we only count image files
            current_image_count = len(image_files)

            if current_image_count < REQUIRED_IMAGES_PER_CLASS:
                additional_images_needed = REQUIRED_IMAGES_PER_CLASS - current_image_count
                print(f"Need to generate {additional_images_needed} more images for class at {output_path}.")

                # Progress bar for additional image generation
                with tqdm(total=additional_images_needed, desc=f"Generating more images in {output_path}") as pbar:
                    while additional_images_needed > 0:
                        with Pool() as pool:
                            # Select files that have not been augmented yet
                            files_to_augment = [f for f in image_files if f not in shared_unique_hashes]
                            # Randomly select a batch of files for augmentation
                            selected_files = random.sample(files_to_augment, min(additional_images_needed, len(files_to_augment)))
                            # Prepare arguments for the process_and_save_augmented function
                            args = [(os.path.join(output_path, f), output_path, shared_unique_hashes, lock, class_prefix, class_counter) for f in selected_files]
                            # Perform the augmentations
                            pool.map(process_and_save_augmented, args)

                            # Update counters and trackers
                            additional_images_needed -= len(selected_files)
                            image_files.update(f"{class_prefix}_{class_counter[class_prefix]:05d}.png" for f in selected_files)
                            pbar.update(len(selected_files))

                print(f"Completed generating images for class at {output_path}.")

if __name__ == '__main__':
    main()
