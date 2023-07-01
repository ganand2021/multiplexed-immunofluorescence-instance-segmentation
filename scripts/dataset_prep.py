import os
import shutil

# Path to the data directory
ROUGH_DATA_DIR = r'D:\multiplexed-immunofluorescence-instance-segmentation\data\rough_data\Vectra'
IMAGE_DATA_DIR = r'D:\multiplexed-immunofluorescence-instance-segmentation\data\final\image'
MASK_DATA_DIR = r'D:\multiplexed-immunofluorescence-instance-segmentation\data\final\mask'

def get_files_wrt_extension(path, file_extension):
    """Get all files with a given extension in the given directory"""
    file_paths = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(file_extension):
                file_paths.append(os.path.join(root, file))
    return file_paths

def copy_files_to_directory(file_list, destination_directory):
    for file_path in file_list:
        _, filename = os.path.split(file_path)
        destination_path = os.path.join(destination_directory, filename)
        os.makedirs(destination_directory, exist_ok=True)
        shutil.copy2(file_path, destination_path)
        renamed_path = os.path.join(destination_directory, filename.lower())
        os.rename(destination_path, renamed_path)
        
def main():
    file_extensions = {
        'img' : '.tif',
        'mask' : '.png',
    }
    image_paths = get_files_wrt_extension(ROUGH_DATA_DIR, file_extensions['img'])
    mask_paths = get_files_wrt_extension(ROUGH_DATA_DIR, file_extensions['mask'])
    
    copy_files_to_directory(image_paths, IMAGE_DATA_DIR)
    copy_files_to_directory(mask_paths, MASK_DATA_DIR)
    
    print("Dataset Prepped")
    
if __name__ == "__main__":
    main()