from sklearn.model_selection import train_test_split
import shutil
import os

# Define source paths for the images
source_path_with_masks = '/Users/markzhuang/Desktop/Project/with_mask_processed'
source_path_without_masks = '/Users/markzhuang/Desktop/Project/without_mask_processed'

# Define destination paths for train, val, and test sets
train_path = '/Users/markzhuang/Desktop/Project/train'
val_path = '/Users/markzhuang/Desktop/Project/val'
test_path = '/Users/markzhuang/Desktop/Project/test'

# Create the destination directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Function to copy files to their respective directories
def copy_files(file_list, source_dir, destination_subfolder):
    destination_dir = os.path.join(destination_subfolder, os.path.basename(source_dir))
    os.makedirs(destination_dir, exist_ok=True)
    for file in file_list:
        shutil.copy2(os.path.join(source_dir, file), destination_dir)

# Get all file names from the source directories, excluding hidden files
files_with_masks = [f for f in os.listdir(source_path_with_masks) if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
files_without_masks = [f for f in os.listdir(source_path_without_masks) if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Print the total number of images in each source folder
print(f"Total number of 'with_mask' images: {len(files_with_masks)}")
print(f"Total number of 'without_mask' images: {len(files_without_masks)}")

# Set a random seed for reproducibility
random_seed = 42

# Split data into train, validation, and test sets
train_files_with_masks, test_val_files_with_masks = train_test_split(
    files_with_masks, test_size=0.3, random_state=random_seed)
val_files_with_masks, test_files_with_masks = train_test_split(
    test_val_files_with_masks, test_size=0.5, random_state=random_seed)

train_files_without_masks, test_val_files_without_masks = train_test_split(
    files_without_masks, test_size=0.3, random_state=random_seed)
val_files_without_masks, test_files_without_masks = train_test_split(
    test_val_files_without_masks, test_size=0.5, random_state=random_seed)

# Copy files to their respective directories
copy_files(train_files_with_masks, source_path_with_masks, train_path)
copy_files(val_files_with_masks, source_path_with_masks, val_path)
copy_files(test_files_with_masks, source_path_with_masks, test_path)

copy_files(train_files_without_masks, source_path_without_masks, train_path)
copy_files(val_files_without_masks, source_path_without_masks, val_path)
copy_files(test_files_without_masks, source_path_without_masks, test_path)

# Print the number of files in each set for 'with_mask'
print(f"'with_mask' Training set: {len(train_files_with_masks)} images")
print(f"'with_mask' Validation set: {len(val_files_with_masks)} images")
print(f"'with_mask' Test set: {len(test_files_with_masks)} images")

# Print the number of files in each set for 'without_mask'
print(f"'without_mask' Training set: {len(train_files_without_masks)} images")
print(f"'without_mask' Validation set: {len(val_files_without_masks)} images")
print(f"'without_mask' Test set: {len(test_files_without_masks)} images")

print("Data split into train, validation, and test sets.")
