import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion, CoarseDropout

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path):
    """ X = Images and Y = masks """

    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "1st_manual", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "1st_manual", "*.gif")))

    return (train_x, train_y), (test_x, test_y)



def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for idx, (x, y) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        """ Extracting names """
        name = x.split("\\")[-1].split(".")[0]

        """ Reading image and mask """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = imageio.mimread(y)[0]
        
        row, col, _ = x.shape
        
        row1 = row // 2
        row2 = row1 // 2
        row3 = (row + row1) // 2
        
        col1 = col // 2
        col2 = col1 // 2
        col3 = (col + col1) // 2

        if augment == True:
            
            img_piece1 = x[0:row1, 0:col1]
            img_piece2 = x[0:row1, col2:col3]
            img_piece3 = x[0:row1, col1:col]
            
            img_piece4 = x[row2:row3, 0:col1]
            img_piece5 = x[row2:row3, col2:col3]
            img_piece6 = x[row2:row3, col1:col]
            
            img_piece7 = x[row1:row, 0:col1]
            img_piece8 = x[row1:row, col2:col3]
            img_piece9 = x[row1:row, col1:col]
            
            mask_piece1 = y[0:row1, 0:col1]
            mask_piece2 = y[0:row1, col2:col3]
            mask_piece3 = y[0:row1, col1:col]
            
            mask_piece4 = y[row2:row3, 0:col1]
            mask_piece5 = y[row2:row3, col2:col3]
            mask_piece6 = y[row2:row3, col1:col]
            
            mask_piece7 = y[row1:row, 0:col1]
            mask_piece8 = y[row1:row, col2:col3]
            mask_piece9 = y[row1:row, col1:col]
            
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = GridDistortion(p=1)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            X = [x, x1, x2, x3, x4, x5, img_piece1, img_piece2, img_piece3, img_piece4, 
                 img_piece5, img_piece6, img_piece7, img_piece8, img_piece9]
            Y = [y, y1, y2, y3, y4, y5, mask_piece1, mask_piece2, mask_piece3, mask_piece4, 
                 mask_piece5, mask_piece6, mask_piece7, mask_piece8, mask_piece9]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, (W, H))
            m = cv2.resize(m, (W, H))

            if len(X) == 1:
                tmp_image_name = f"{name}.jpg"
                tmp_mask_name = f"{name}.jpg"
            else:
                tmp_image_name = f"{name}_{index}.jpg"
                tmp_mask_name = f"{name}_{index}.jpg"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)

    """ Load the data """
    data_path = "C:\\Users\Administrator\Desktop\\Unet_PYtorch"
    (train_x, train_y), (test_x, test_y) = load_data(data_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")

    """ Creating directories """
    create_dir("new_data/train/image")
    create_dir("new_data/train/mask")
    create_dir("new_data/test/image")
    create_dir("new_data/test/mask")

    augment_data(train_x, train_y, "new_data/train/", augment=True)
    augment_data(test_x, test_y, "new_data/test/", augment=False)
