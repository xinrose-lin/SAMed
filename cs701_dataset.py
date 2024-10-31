import h5py
import numpy as np
from PIL import Image
import glob
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
### compile val dataset into 
### .npy.h5 for each volume

# List of image file paths (e.g., 'images/*.png' if they're in the 'images' folder)
# filebase = 'public_leaderboard_data/


def process_data_npy_h5():
    images_dir = 'train_images'
    labels_dir = 'train_labels'
    volume_names = os.listdir(f'public_leaderboard_data/{images_dir}')
    for volume_name in volume_names: 
        # print(volume_name)
        if volume_name == '.DS_Store':
            continue
        image_files = sorted(glob.glob(f'public_leaderboard_data/{images_dir}/{volume_name}/*.png'))  # Ensure sorted order

        images = []
        for file in image_files:
            img = Image.open(file)
            img_array = np.array(img)  # Convert to numpy array
            images.append(img_array)
        images_np = np.stack(images, axis=0)
        # print(images_np.shape)
        # print(volume_name)

        label_files = sorted(glob.glob(f'public_leaderboard_data/{labels_dir}/{volume_name}/*.png'))  # Ensure sorted order

        labels = []
        for file in label_files:
            img = Image.open(file)
            img_array = np.array(img)  # Convert to numpy array
            labels.append(img_array)
        labels_np = np.stack(labels, axis=0)
        # print(labels_np.shape)
        # print(volume_name)

        assert labels_np.shape == images_np.shape

        with h5py.File(f'public_leaderboard_data/vol-data-npy/train/{volume_name}.npy.h5', 'w') as h5_file:
            h5_file.create_dataset('images', data=images_np)
            h5_file.create_dataset('labels', data=labels_np)  # Add labels dataset

        print(f"Saved images and labels to {volume_name}.npy.h5. {images_np.shape, labels_np.shape}")


    # with h5py.File(f'public_leaderboard_data/vol-data-npy/train/{volume_name}.npy.h5', 'r') as f:
    #     # List all groups
    #     print(list(f.keys()))
    #     # Access a specific dataset
    #     dataset_image = f['images'][:]
    #     print(dataset_image.shape, type(dataset_image))
    #     dataset_label = f['labels'][:]
    #     print(dataset_label.shape, type(dataset_label))
    # break


class cs701_dataset(Dataset):
    def __init__(self, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        
        self.val_data = glob.glob('public_leaderboard_data/vol-data-npy/val/*')
        self.train_data = glob.glob('public_leaderboard_data/vol-data-npy/train/*')
        self.full_data_list = self.train_data + self.val_data
        # self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        # self.data_dir = base_dir # public_leaderboard_data/vol-data-npy
        # self.list_dir = list_dir

        if self.split == "train":
            self.indices = range(len(self.train_data))  # Indices for training
        elif self.split == 'val':
            self.indices = range(len(self.train_data), len(self.full_data_list))  # Indices for testing

    def __len__(self):
        # TODO: glob entire directory
        return len(self.full_data_list)

    def __getitem__(self, idx):

        if self.split == "train":
            filename = self.full_data_list[self.indices[idx]]
            with h5py.File(filename, 'r') as f:
                # Access a specific dataset
                image = f['images'][:]
                # print('train data', filename, image.shape, type(image))
                label = f['labels'][:]
                # print(label.shape, type(label))
        elif self.split == "val":
            filename = self.full_data_list[self.indices[idx]]
            with h5py.File(filename, 'r') as f:
                image = f['images'][:]
                # print('val data', filename, image.shape, type(image))
                label = f['labels'][:]
                # print(label.shape, type(label))

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.full_data_list[idx]
        return sample


if __name__ == '__main__':
    cs701_testds = cs701_dataset(split='val')
    # db_test = cs701_ds(base_dir=args.volume_path, list_dir=args.list_dir, split='val')
    print('testds done')
    testloader = DataLoader(cs701_testds, batch_size=1, shuffle=False, num_workers=1)
    print('test loader')
    for batch in testloader: 
        print(batch.keys())
        break
    
# why return twice
# ## val data public_leaderboard_data/vol-data-npy/val/42.npy.h5 (100, 512, 512) <class 'numpy.ndarray'>
# (100, 512, 512) <class 'numpy.ndarray'>
# val data public_leaderboard_data/vol-data-npy/val/43.npy.h5 (95, 512, 512) <class 'numpy.ndarray'>
# dict_keys(['image', 'label', 'case_name'])
# (95, 512, 512) <class 'numpy.ndarray'>

# label_files = sorted(glob.glob('path/to/images/*.png'))  # Ensure sorted order
# images = []
# # Save the numpy array to an HDF5 file
# with h5py.File('output_images_with_labels.npy.h5', 'w') as h5_file:
#     h5_file.create_dataset('images', data=images_np)
#     h5_file.create_dataset('labels', data=labels)  # Add labels dataset

# print("Saved images and labels to output_images_with_labels.npy.h5.")


# import numpy as np

# data = np.load('testset/test_vol_h5/case0002.npy.h5')
# print(type(data), data.shape)

# import h5py

# with h5py.File('testset/test_vol_h5/case0002.npy.h5', 'r') as f:
#     # List all groups
#     print(list(f.keys()))
#     # Access a specific dataset
#     dataset_image = f['image'][:]
#     print(dataset_image.shape, type(dataset_image))
#     dataset_label = f['label'][:]
#     print(dataset_label.shape, type(dataset_label))

# # Load the .npy file
# train_img = np.load(train_images_path)
# train_labels = np.load(train_labels_path)
# val_img = np.load(val_images_path)
# val_labels = np.load(val_labels_path)

# print('train_img', train_img.shape)
# print('train_labels', train_labels.shape)
# print('val_img', val_img.shape)
# print('val_labels', val_labels.shape)


