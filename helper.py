from skimage.metrics import structural_similarity
import cv2
import os
import argparse
import math
import numpy as np

def psnr_helper(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def outer_helper(hr_dir, sr_dir, scale, ssim=True, psnr=True):
    sr_files = os.listdir(sr_dir)

    ssim = 0
    psnr = 0
    for file in sr_files:
        # file = sr_files[0]
        sr = f"{sr_dir}/{file}"
        print(sr)

        hr = f"{hr_dir}/{file.replace(f'_x{scale}_SR.png', '.png')}"
        # print(hr)

        original = cv2.imread(sr,0)
        contrast = cv2.imread(hr,0)


    # if ssim:
        (score, diff) = structural_similarity(original, contrast, win_size=7, full=True)
        diff = (diff * 255).astype("uint8")

        print("SSIM:{:2f}".format(score))
        ssim += score

    # if psnr:
        score= psnr_helper(original, contrast)

        print("PSNR:{:2f}".format(score))
        psnr += score
# if ssim:
    print("Average SSIM={:2f}".format(ssim / len(sr_files)))
# if psnr:
    print("Average PSNR={:2f}".format(psnr / len(sr_files)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hr', type=str, default="/home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_HR")
    parser.add_argument('--sr', type=str, required=True)
    parser.add_argument('--scale', type=str, required=True)
    args = parser.parse_args()
    
    outer_helper(args.hr, args.sr, args.scale)