# DLSRProject
Source code of IMDMKET and FEIMDN

## Preparation of Dataset
1. ### Download raw dataset
      1. Please download raw IXI-T2 dataset at http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T2.tar
      2. For folder `Sampled_IXI_T2`, change the name into `IXI_T2` and remove `.placehold` file
      3. Extract downloaded tar file into `IXI_T2` folder
      4. Eventurally, there will be a set of *.nii.gz files inside `IXI_T2` folder.
2. ### Preprocessing dataset and generating PNG for HR and LR images
    1. For folder `Sampled_IXI`, change the name into  `IXI`, which is our target folder of PNG generation. (Note: The name must be `IXI`!!) Remove all `.placeholder` files in subfolders.
    2. Within `src`, run `png_prepare.py` with full command:
        ```
            python3 {project_path}/src/png_helper.py \
                    --x2 --x4 --x8 --genehr \
                    --source_dir {project_path}/IXI-T2 \
                    --target_dir {project_path}/IXI
        ```
        where option `--x2`, `--x4` and `--x8` are used for generating corresponding bicubic downsampled LR png; `--genehr` is to generate HR png.
    3. The files `png_helper.py` and `png.sh` are acutal file that I used with UQ Cluster cloud server.



<br/><br/>


## Models of IMDMKET, FEIMDN and historical networks
### IMDMKET
Inside `src\model`, `IMDMKET7.py` are final version of IMDMKET. Exampled command for running IMDMKET with scale factor 2: 
```
    python3 {project_path}\src\main.py $* \
                --scale 2 \
                --epochs 200 \
                --lr 0.0002 \
                --save imdmket7_x2_lr00002_g75 \
                --model IMDMKET7 \
                --decay 20-40-60-80-100-120-140-160-180 \
                --gamma 0.75 \
                --batch_size 2 \
                --save imdn_x2_epochs400_lr00001_vgpu40 \
                --patch_size 256 \
                --dir_data {project_path} \
                --data_train IXI \
                --data_test IXI \
                --data_range 1-28157/28158-31677 \
                --test_every 1 \
                --n_colors 1 \
                --reset \
                --no_augment
```
- `imdm*.py` are the historical IMDMKET models without Transformer backbone (efficient transformer, ET) including the models chosen for ablation study.
- `imdmket*.py` are the historical IMDMKET models with Transformer backbone (efficient transformer, ET).

### FEIMDN
Inside `src\model`, `net22.py` are final version of FEIMDN. Exampled command for running FEIMDN with scale factor 4: 
```
    python3 {project_path}\src\main.py $* \
                --scale 4 \
                --epochs 200 \
                --lr 0.0002 \
                --save net22_x4_epoch200_lr00002_vgpu40_d80_160 \
                --model NET22 \
                --decay 80-160 \
                --gamma 0.75 \
                --n_brmblocks 6 \
                --patch_size 256 \
                --dir_data {project_path} \
                --data_train IXI \
                --data_test IXI \
                --data_range 1-28157/28158-31677 \
                --test_every 1 \
                --n_colors 1 \
                --reset \
                --no_augment
```
where n_brmblocks indicates the number of frequency-based mapping blocks (FMBs) that actually used.
- `net*.py` are the historical FEIMDN models.

<br/><br/>

### Note：
Since the project is actually trained on UQ cluster server, there are bunch of shell files which is I used for training or testing models.  
Within `{project_path}\src\`:
- `setupenv.sh`: configurating the environment for necessary library.
- For training and testing model:

    | Sbatch file |   Goal   | Cluster Node |
    |:-----------:|:--------:|:------------:|
    |    tr.sh    | training |    vgpu40    |
    |   tr10.sh   | training |    vgpu10    |
    |   tr20.sh   | training |    vgpu20    |
    |   train.sh  | training |     p100     |
    |    te.sh    |  testing |    vgpu40    |
    |   test.sh   |  testing |     p100     |
    
In addition, within `{project_path}`:
- `helper.py` and `h.sh` are used to calculate SSIM score. 
- `bicubic.py` and `b.sh` are used to generate the baseline result of bicubic upsampling method.
