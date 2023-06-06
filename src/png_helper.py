import argparse
import os
import random
import SimpleITK as sitk
import cv2


def png_helper(origin_path, hr_png_path, lr_png_path, args):
    if not os.path.exists(hr_png_path):
        os.mkdir(hr_png_path)
    if not os.path.exists(lr_png_path):
        os.mkdir(lr_png_path)
    
    files = os.listdir(origin_path)
    random.shuffle(files)
    
    for filename in files:
        pure_filename = filename.replace(".nii.gz", "")
        # print(pure_filename)
        img_src = sitk.ReadImage(f"{origin_path}/{filename}")
        img_array = sitk.GetArrayViewFromImage(img_src)

        if img_array.shape[0] < 100:
            continue

        for i in range(30, 91):
            img_slice = img_array[i,:,:]
            # Normalization
            hr_img = (((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())) * 255).astype("uint8")
            if args.genehr:
                cv2.imwrite(f"{hr_png_path}/{pure_filename}_{i}.png", hr_img)
            
            #Downsample image x{scale}
            lr_img_x2 = cv2.resize(hr_img, (0,0), fx=1/args.scale, fy=1/args.scale, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{lr_png_path}/{pure_filename}_{i}x{args.scale}.png", lr_img_x2)
            
    print("Finish png generation")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--genehr', action='store_true')
    args = parser.parse_args()
    
    png_helper("/home/Student/s4427443/IXI-T2", 
                "../MRI/DIV2K/DIV2K_train_HR", 
                f"../MRI/DIV2K/DIV2K_train_LR_bicubic/X{args.scale}", 
                args)