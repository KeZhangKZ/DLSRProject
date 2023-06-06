import cv2
import os
import argparse
import math
import numpy as np


def bic(args):
    hr = "/home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_HR"
    lr2 = "/home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_LR_bicubic/X2"
    lr4 = "/home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_LR_bicubic/X4"
    lr8 = "/home/Student/s4427443/edsr40/MRI/DIV2K/DIV2K_train_LR_bicubic/X8"
    # sr = "/home/Student/s4427443/edsr40/experiment/test_edsr_x2/results-DIV2K"
    bic2 = "/home/Student/s4427443/edsr40/experiment/test_bicubic_x2/"
    bic4 = "/home/Student/s4427443/edsr40/experiment/test_bicubic_x4/"
    bic8 = "/home/Student/s4427443/edsr40/experiment/test_bicubic_x8/"
    if not os.path.exists(bic2):
        os.mkdir(bic2)
    if not os.path.exists(bic4):
        os.mkdir(bic4)
    if not os.path.exists(bic8):
        os.mkdir(bic8)
    begin = 31678
    end = 35197
    # sr_files = os.listdir(sr)
    # "IXI587-Guys-1128-T2_48_x2_SR.png"
    # "x2.png"
    files = sorted(os.listdir(hr))[begin - 1 : end]
    for file in files:
        # if not file.replace(".png", "_x2_SR.png") in sr_files:
        #     print("Fail")
        #     return

        #Upsample image x{scale}
        if args.x2:
            l2 = f"{lr2}/{file.replace('.png', 'x2.png')}"
            l2_img = cv2.imread(l2)
            sr_x2 = cv2.resize(l2_img, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{bic2}/{file.replace('.png', '_x2_SR.png')}", sr_x2)

        if args.x4:
            l4 = f"{lr4}/{file.replace('.png', 'x4.png')}"
            l4_img = cv2.imread(l4)
            sr_x4 = cv2.resize(l4_img, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{bic4}/{file.replace('.png', '_x4_SR.png')}", sr_x4)

        if args.x8:
            l8 = f"{lr8}/{file.replace('.png', 'x8.png')}"
            l8_img = cv2.imread(l8)
            sr_x8 = cv2.resize(l8_img, (0,0), fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{bic8}/{file.replace('.png', '_x8_SR.png')}", sr_x8)

    print("Success")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--x8', action='store_true')
    parser.add_argument('--x4', action='store_true')
    parser.add_argument('--x2', action='store_true')
    args = parser.parse_args()
    bic(args)
