import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate

# when the data augumentaution is performed to the images it should be saved in a separate folder
# This function creates the folder for augumented image
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#Data Augumentation function
def augument_data(images, masks, save_path, augument=True):
    size = (512, 512)

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        # Extracting the name of the file
        name = x.split("\\")[-1].split(".")[0]
        
        # Reading image and mask
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]

        # print(x.shape)
        # print(y.shape)
        if augument == True:
            #Horizontal Flip
            aug = HorizontalFlip(p=1.0) # p=probabelity of applying data augmentation
            augmented = aug(image=x, mask=np.expand_dims(y, axis=-1))
            x1 = augmented["image"]
            y1 = augmented["mask"][:, :, 0]

            # Vertical Flip
            aug = VerticalFlip(p=1.0) # p=probabelity of applying data augmentation
            augmented = aug(image=x, mask=np.expand_dims(y, axis=-1))
            x2 = augmented["image"]
            y2 = augmented["mask"][:, :, 0]

            # Rotate
            aug = Rotate(limit=45, p=1.0) # p=probabelity of applying data augmentation
            augmented = aug(image=x, mask=np.expand_dims(y, axis=-1))
            x3 = augmented["image"]
            y3 = augmented["mask"][:, :, 0]

            X = [x, x1, x2, x3]
            Y = [y, y1, y2, y3]
            
        else:
            X = [x]
            Y = [y]

        index = 0
        for i,m in zip(X, Y):
            i = cv2.resize(i, size)
            m = cv2.resize(m, size)

            temp_image_name = f"{name}_{index}.png"
            temp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", temp_image_name)
            mask_path = os.path.join(save_path, "mask", temp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index+=1

def load_data(path):
    # Training images
    train_X = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    # Testing images
    test_X = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_X, train_y), (test_X, test_y)

if __name__ == "__main__":
    # SEEDING
    np.random.seed(42)

    # Load the data
    data_path = "Dataset"
    (train_X, train_y), (test_X, test_y) = load_data(data_path)

    # printing the no. of training and testing part
    print(f"Train: {len(train_X)} - {len(train_y)}")
    print(f"Test: {len(test_X)} - {len(test_y)}")

    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    #performing Data augumentation
    augument_data(train_X, train_y, "new_data/train/", augument=True)
    augument_data(train_X, train_y, "new_data/test/", augument=False)