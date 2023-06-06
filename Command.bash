####################Training####################

################X2################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/rdn_x2_epochs200_lr00001.out --job-name rdntr tr.sh --scale 2 --epochs 200 --lr 0.0001 --save rdn_x2_epochs200_lr00001 --model RDN

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/fsrcnn_x2_epochs400_lr00001.out --job-name ffftr tr.sh --scale 2 --epochs 400 --lr 0.0001 --save fsrcnn_x2_epochs400_lr00001 --model FSRCNN

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/rdn_x2_epochs200_lr00001_cpu8.out --job-name rdntr2cpu8 tr.sh --scale 2 --epochs 200 --lr 0.0001 --save rdn_x2_epochs200_lr00001_cpu8 --model RDN --batch_size 8


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/rdn_x2_epochs200_lr00001_cpu8_2.out --job-name rdntr2cpu8_2 train.sh --scale 2 --epochs 200 --lr 0.0001 --save rdn_x2_epochs200_lr00001_cpu8_2 --model RDN


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/rdn_x2_epochs200_lr000001.out --job-name rdntr25 tr.sh --scale 2 --epochs 200 --lr 0.00001 --save rdn_x2_epochs200_lr000001 --model RDN --batch_size 8

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_vgpu40.out --job-name ebrn2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_vgpu40 --model EBRN

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_vgpu20.out --job-name ebrn2_20 tr20.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_vgpu20 --model FSRCNN --n_brmblocks 1 --batch_size 4 --n_threads 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_p100.out --job-name ebrn2_100 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_p100 --model EBRN --n_brmblocks 2 --batch_size 4 --n_threads 4 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_p1000_b4_t4.out --job-name ebrn2_100 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_p100 --model EBRN --n_brmblocks 4 --batch_size 4 --n_threads 4 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_p100_bk4_b16.out --job-name ebrn2_100_bk4_b16 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_p100_bk4_b16 --model EBRN --n_brmblocks 4 --n_threads 4 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_p100_bk4_b8.out --job-name ebrn2_100_bk4_b8 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_p100_bk4_b8 --model EBRN --n_brmblocks 4 --batch_size 8 --n_threads 4 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_p100_bk2_b16.out --job-name ebrn2_100_bk2_b16 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_p100_bk2_b16 --model EBRN --n_brmblocks 2 --n_threads 4 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_p100_bk2_b8.out --job-name ebrn2_100_bk2_b8 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_p100_bk2_b8 --model EBRN --n_brmblocks 2 --batch_size 8 --n_threads 4 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x2_epochs400_lr00001_p100_bk2_b8.out --job-name ebrn2_100_bk2_b8 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save ebrn_x2_epochs400_lr00001_p100_bk2_b8 --model EBRN --n_brmblocks 2 --batch_size 8 --n_threads 4 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/t_x2_epochs400_lr00001_vgpu40.out --job-name t_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save t_x2_epochs400_lr00001_vgpu40 --model EBRN

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/t_x2_epochs400_lr00001.out --job-name t tr.sh --scale 2 --epochs 400 --lr 0.0001 --save t_x2_epochs400_lr00001 --model FSRCNN

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/mt.out --job-name mt mttr.sh

(Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions)



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net1_x2_epochs400_lr00001_vgpu40_final.out --job-name net1_2_40_f tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net1_x2_epochs400_lr00001_vgpu40_final --model NET1

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net1_x2_epochs400_lr00001_vgpu10.out --job-name net1_2_10 tr10.sh --scale 2 --epochs 400 --lr 0.0001 --save net1_x2_epochs400_lr00001_vgpu10 --model NET1

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net1_x2_epochs400_lr00001_vgpu10_t4_b4.out --job-name net1_2_10_t4_b4 tr10.sh --scale 2 --epochs 400 --lr 0.0001 --save net1_x2_epochs400_lr00001_vgpu10_t4_b4 --model NET1 --n_threads 4 --batch_size 4



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net1_x2_epochs400_lr00001_p100.out --job-name net1_100 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save net1_x2_epochs400_lr00001_p100 --model NET1 --n_GPUs 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net1_x2_epochs400_lr00001_p100_t2.out --job-name net1_100_t2 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save net1_x2_epochs400_lr00001_p100_t2 --model NET1 --n_GPUs 2  --n_threads 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net1_x4_epochs400_lr00001_p100_t4.out --job-name net1_4_100_t4 trainp100.sh --scale 4 --epochs 400 --lr 0.0001 --save net1_x4_epochs400_lr00001_p100_t4 --model NET1 --n_GPUs 2  --n_threads 4




##################### NET2 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net2_x2_epochs400_lr00001_vgpu40.out --job-name net2_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net2_x2_epochs400_lr00001_vgpu40 --model NET2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net2f_x2_epochs400_lr00001_vgpu40.out --job-name net2f_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net2f_x2_epochs400_lr00001_vgpu40 --model NET2F



################################   X4   ################################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/fsrcnn_x4_epochs400_lr00001_vgpu20.out --job-name ffftr4_20 tr20.sh --scale 4 --epochs 500 --lr 0.0001 --save fsrcnn_x4_epochs400_lr00001_vgpu20 --model FSRCNN --pre_train ../experiment/fsrcnn_x2_epochs400_lr00001/model/model_best.pt

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/fsrcnn_x4_epochs400_lr00001.out --job-name ffftr4 tr.sh --scale 4 --epochs 400 --lr 0.0001 --save fsrcnn_x4_epochs400_lr00001 --model FSRCNN --pre_train ../experiment/fsrcnn_x2_epochs400_lr00001/model/model_best.pt



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/edsr_x4_epochs400_lr00001_vgpu20.out --job-name edsrtr4_20 tr20.sh --scale 4 --epochs 300 --lr 0.0001 --save edsr_x4_epochs300_lr00001_vgpu20 --model EDSR --pre_train ../experiment/edsr_baseline_x2_epoch300_lr/model/model_best.pt

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/edsr_x4_epochs400_lr00001.out --job-name edsrtr4 tr.sh --scale 4 --epochs 300 --lr 0.0001 --save edsr_x4_epochs300_lr00001 --model EDSR --pre_train ../experiment/edsr_baseline_x2_epoch300_lr/model/model_best.pt



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/rdn_x4_epochs200_lr0000001.out --job-name rdntr4 tr.sh --scale 4 --epochs 200 --lr 0.000001 --save rdn_x4_epochs200_lr0000001 --model RDN --batch_size 4 --pre_train ../experiment/rdn_x2_epochs200_lr000001/model/model_best.pt



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x4_epochs400_lr00001_vgpu40_bk4_b16.out --job-name ebrn4_40 tr.sh --scale 4 --epochs 400 --lr 0.0001 --save ebrn_x4_epochs400_lr00001_vgpu40 --model EBRN --pre_train ../experiment/ebrn_x2_epochs400_lr00001_vgpu40/model/model_best.pt

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x4_epochs400_lr00001_vgpu40_bk4_b16.out --job-name ebrn4_40 tr.sh --scale 4 --epochs 400 --lr 0.0001 --save ebrn_x4_epochs400_lr00001_vgpu40 --model EBRN --pre_train ../experiment/ebrn_x2_epochs400_lr00001_vgpu40/model/model_best.pt

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x4_epochs400_lr00001_vgpu40_bk4_b16_nop.out --job-name ebrn4_40_nop tr.sh --scale 4 --epochs 400 --lr 0.0001 --save ebrn_x4_epochs400_lr00001_vgpu40_nop --model EBRN

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x4_epochs400_lr00001_vgpu40_bk4_b16_nop.out --job-name ebrn4_40_nop tr.sh --scale 4 --epochs 400 --lr 0.0001 --save ebrn_x4_epochs400_lr00001_vgpu40_nop --model EBRN


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x4_epochs400_lr00001_p100_bk4_b16.out --job-name ebrn4_100 trainp100.sh --scale 4 --epochs 400 --lr 0.0001 --save ebrn_x4_epochs400_lr00001_p100_bk4_b16 --model EBRN --pre_train ../experiment/ebrn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --n_threads 4 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x4_epochs400_lr00001_p100_bk4_b16_nop.out --job-name ebrn4_100_nop trainp100.sh --scale 4 --epochs 400 --lr 0.0001 --save ebrn_x4_epochs400_lr00001_p100_bk4_b16_nop --model EBRN --n_threads 4 --n_GPUs 2


##################### NET2 ##################### low 3c high 2mkam 
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net2_x2_epochs400_lr00001_vgpu40.out --job-name net2_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net2_x2_epochs400_lr00001_vgpu40 --model NET2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net2f_x4_epochs400_lr00001_vgpu40.out --job-name net2f_4_40 tr.sh --scale 4 --epochs 400 --lr 0.0001 --save net2f_x4_epochs400_lr00001_vgpu40 --model NET2F


##################### NET3 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net3_x2_epochs400_lr00001_vgpu40.out --job-name net3_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net3_x2_epochs400_lr00001_vgpu40 --model NET3

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net2f_x4_epochs400_lr00001_vgpu40.out --job-name net2f_4_40 tr.sh --scale 4 --epochs 400 --lr 0.0001 --save net2f_x4_epochs400_lr00001_vgpu40 --model NET2F




##################### NET4 ##################### low 3c high 2mkam dense
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net4_x2_epochs400_lr00001_vgpu40.out --job-name net4_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net4_x2_epochs400_lr00001_vgpu40 --model NET4



##################### NET5 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net5_x2_epochs400_lr00001_vgpu40.out --job-name net5_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net5_x2_epochs400_lr00001_vgpu40 --model NET5


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net5_x2_epochs400_lr00001_p100_b16_t8.out --job-name net5_2_100_t8 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save net5_x2_epochs400_lr00001_p100_b16_t8 --model NET5 --n_threads 8 --n_GPUs 2



##################### NET6 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net6_x2_epochs400_lr00001_vgpu40.out --job-name net6_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net6_x2_epochs400_lr00001_vgpu40 --model NET6


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net6_x2_epochs400_lr00001_p100_b16_t8.out --job-name net6_2_100_t8 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save net6_x2_epochs400_lr00001_p100_b16_t8 --model NET6 --n_threads 8 --n_GPUs 2



##################### NET7 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net7_x2_epochs400_lr00001_vgpu40.out --job-name net7_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save net7_x2_epochs400_lr00001_vgpu40 --model NET7


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net7_x2_epochs400_lr00001_p100_b16_t8.out --job-name net7_2_100_t8 trainp100.sh --scale 2 --epochs 400 --lr 0.0001 --save net7_x2_epochs400_lr00001_p100_b16_t8 --model NET7 --n_threads 8 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net7_x2_epochs400_lr0001_p100_d10.out --job-name n7_2_100_t8 trainp100.sh --scale 2 --epochs 200 --lr 0.001 --save net7_x2_epochs400_lr0001_p100_d10 --model NET7 --n_threads 8 --n_GPUs 2  --decay 50









##################### NET8 #####################
############## train ################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net8_x2_epoch200_lr00002_p100_d100.out --job-name n8_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net8_x2_epoch200_lr00002_p100_d100 --model NET8 --decay 100 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net8_x2_epoch200_lr00002_vgpu40_d100.out --job-name n8_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net8_x2_epoch200_lr00002_vgpu40_d100 --model NET8 --decay 100 --n_brmblocks 6



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2







##################### NET9 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn_x2_epochs400_lr00001_vgpu40.out --job-name imdn_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0005 --save imdn_x2_epochs400_lr00001_vgpu40 --model IMDN



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net9_x2_epoch200_lr00005_vgpu40_d20_50_100.out --job-name n9_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0005 --save net9_x2_epoch200_lr00005_vgpu40_d20_50_100 --model NET9 --decay 20-50-100 --n_threads 6 --n_brmblocks 4



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net9_x2_epoch200_lr00002_vgpu40_d100.out --job-name n9_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net9_x2_epoch200_lr00002_vgpu40_d100 --model NET9 --decay 100 --n_brmblocks 4



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2









##################### NET10 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net10_x2_epoch200_lr00002_p100_d100.out --job-name n10_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net10_x2_epoch200_lr00002_p100_d100 --model NET10 --decay 100 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net10_x2_epoch200_lr00002_vgpu40_d100.out --job-name n10_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net10_x2_epoch200_lr00002_vgpu40_d100 --model NET10 --decay 100 --n_brmblocks 6



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2







##################### NET11 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net11_x2_epoch200_lr00002_p100_d80_160.out --job-name n11_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net11_x2_epoch200_lr00002_p100_d80_160 --model NET11 --decay 80-160 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net11_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n11_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net11_x2_epoch200_lr00002_vgpu40_d80_160 --model NET11 --decay 80-160



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2






##################### NET12 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net12_x2_epoch200_lr00002_p100_d80_160.out --job-name n12_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net12_x2_epoch200_lr00002_p100_d80_160 --model NET12 --decay 80-160 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net12_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n12_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net12_x2_epoch200_lr00002_vgpu40_d80_160 --model NET12 --decay 80-160



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2







##################### NET13 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net13_x2_epoch200_lr00002_p100_d80_160.out --job-name n13_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net13_x2_epoch200_lr00002_p100_d80_160 --model NET13 --decay 80-160 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net13_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n13_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net13_x2_epoch200_lr00002_vgpu40_d80_160 --model NET13 --decay 80-160



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2







##################### NET14 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net14_x2_epoch200_lr00002_p100_d80_160.out --job-name n14_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net14_x2_epoch200_lr00002_p100_d80_160 --model NET14 --decay 80-160 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net14_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n14_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net14_x2_epoch200_lr00002_vgpu40_d80_160 --model NET14 --decay 80-160



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2







##################### NET15 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net15_x2_epoch200_lr00002_p100_d80_160.out --job-name n15_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net15_x2_epoch200_lr00002_p100_d80_160 --model NET15 --decay 80-160 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net15_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n15_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net15_x2_epoch200_lr00002_vgpu40_d80_160 --model NET15 --decay 80-160



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2






##################### NET17 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net17_x2_epoch200_lr00002_p100_d80_160.out --job-name n17_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net17_x2_epoch200_lr00002_p100_d80_160 --model NET17 --decay 80-160 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 --gamma 0.75


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net17_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n17_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net17_x2_epoch200_lr00002_vgpu40_d80_160 --model NET17 --decay 80-160 --gamma 0.75



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2







##################### NET19 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net19_x2_epoch200_lr00002_p100_d80_160.out --job-name n19_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net19_x2_epoch200_lr00002_p100_d80_160 --model NET19 --decay 80-160 --n_threads 6 --n_GPUs 2 --n_brmblocks 6 --gamma 0.75


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net19_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n19_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net19_x2_epoch200_lr00002_vgpu40_d80_160 --model NET19 --decay 80-160 --gamma 0.75



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2







##################### NET20 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net20_x2_epoch200_lr00002_p100_d80_160.out --job-name n20_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net20_x2_epoch200_lr00002_p100_d80_160 --model NET20 --decay 80-160 --n_threads 8 --n_GPUs 2 --n_brmblocks 4 --gamma 0.75


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net20_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n20_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net20_x2_epoch200_lr00002_vgpu40_d80_160 --model NET20 --decay 80-160 --gamma 0.75



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2





##################### NET21 #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net21_x2_epoch200_lr00002_p100_d80_160.out --job-name n21_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save net21_x2_epoch200_lr00002_p100_d80_160 --model NET21 --decay 80-160 --n_threads 8 --n_GPUs 2 --n_brmblocks 4 --gamma 0.75

###### 32.441 @epoch 1 ######
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net21_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n21_2 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net21_x2_epoch200_lr00002_vgpu40_d80_160 --model NET21 --decay 80-160 --gamma 0.75 --n_brmblocks 4 --n_feats 48

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_net21_x2_epoch200_lr00002_p100.out --job-name net21te2 test.sh --model NET21 --scale 2 --pre_train ../experiment/net21_x2_epoch200_lr00002_vgpu40_d80_160/model/model_best.pt --save test_net21_x2_epoch200_lr00002_p100 --n_threads 8 --n_GPUs 2 --n_brmblocks 4 --n_feats 48


###### 25.001 @epoch 1 ######
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net21_x4_epoch200_lr00002_vgpu40_d80_160.out --job-name n21_4 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save net21_x4_epoch200_lr00002_vgpu40_d80_160 --model NET21 --decay 80-160 --gamma 0.75 --n_brmblocks 4 --n_feats 48

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_net21_x4_epoch200_lr00002_p100.out --job-name net21te4 test.sh --model NET21 --scale 4 --pre_train ../experiment/net21_x4_epoch200_lr00002_vgpu40_d80_160/model/model_best.pt --save test_net21_x4_epoch200_lr00002_p100 --n_threads 8 --n_GPUs 2 --n_brmblocks 4 --n_feats 48


###### 20.830 @epoch 1 ######
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net21_x8_epoch200_lr00002_vgpu40_d80_160.out --job-name n21_8 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save net21_x8_epoch200_lr00002_vgpu40_d80_160 --model NET21 --decay 80-160 --gamma 0.75 --n_brmblocks 4 --n_feats 48

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_net21_x8_epoch200_lr00002_p100.out --job-name net21te8 test.sh --model NET21 --scale 8 --pre_train ../experiment/net21_x8_epoch200_lr00002_vgpu40_d80_160/model/model_best.pt --save test_net21_x8_epoch200_lr00002_p100 --n_threads 8 --n_GPUs 2 --n_brmblocks 4 --n_feats 48



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2





##################### NET22 #####################
############## train ################

###### 32.441 @epoch 1 ######   36.612
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net22_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name n22_2 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save net22_x2_epoch200_lr00002_vgpu40_d80_160 --model NET22 --decay 80-160 --gamma 0.75 --n_brmblocks 6

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_net22_x2_epoch200_lr00002_p100.out --job-name net22te2 test.sh --model NET22 --scale 2 --pre_train ../experiment/net22_x2_epoch200_lr00002_vgpu40_d80_160/model/model_best.pt --save test_net22_x2_epoch200_lr00002_p100 --n_threads 8 --n_GPUs 2 --n_brmblocks 6


###### 25.001 @epoch 1 ######   27.493
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net22_x4_epoch200_lr00002_vgpu40_d80_160.out --job-name n22_4 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save net22_x4_epoch200_lr00002_vgpu40_d80_160 --model NET22 --decay 80-160 --gamma 0.75 --n_brmblocks 6

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_net22_x4_epoch200_lr00002_p100.out --job-name net22te4 test.sh --model NET22 --scale 4 --pre_train ../experiment/net22_x4_epoch200_lr00002_vgpu40_d80_160/model/model_best.pt --save test_net22_x4_epoch200_lr00002_p100 --n_threads 8 --n_GPUs 2 --n_brmblocks 6


###### 20.830 @epoch 1 ######   23.190
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/net22_x8_epoch200_lr00002_vgpu40_d80_160.out --job-name n22_8 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save net22_x8_epoch200_lr00002_vgpu40_d80_160 --model NET22 --decay 80-160 --gamma 0.75 --n_brmblocks 6

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_net22_x8_epoch200_lr00002_p100.out --job-name net22te8 test.sh --model NET22 --scale 8 --pre_train ../experiment/net22_x8_epoch200_lr00002_vgpu40_d80_160/model/model_best.pt --save test_net22_x8_epoch200_lr00002_p100 --n_threads 8 --n_GPUs 2 --n_brmblocks 6


sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_net22_2.out --job-name ssim_net22_2 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_net22_x2_epoch200_lr00002_p100/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_net22_4.out --job-name ssim_net22_4 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_net22_x4_epoch200_lr00002_p100/results-DIV2K --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_net22_8.out --job-name ssim_net22_8 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_net22_x8_epoch200_lr00002_p100/results-DIV2K --scale 8







##################### IMDN #####################
############## train ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn_x2_epochs400_lr00001_vgpu40.out --job-name imdn_2_40 tr.sh --scale 2 --epochs 400 --lr 0.0001 --save imdn_x2_epochs400_lr00001_vgpu40 --model IMDN


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn_x2_epoch200_lr00002_p100_d80_160.out --job-name i1_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn1_x2_epoch200_lr00002_p100_d80_160 --model IMDN --decay 80-160 --n_threads 8 --n_GPUs 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn_x4_epoch200_lr00002_p100_d80_160.out --job-name i1_4_80160 trainp100.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn1_x4_epoch200_lr00002_p100_d80_160 --model IMDN --decay 80-160 --n_threads 8 --n_GPUs 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn_x8_epoch200_lr00002_p100_d80_160.out --job-name i1_8_80160 trainp100.sh --scale 8 --epochs 200 --lr 0.0002 --save imdn1_x8_epoch200_lr00002_p100_d80_160 --model IMDN --decay 80-160 --n_threads 8 --n_GPUs 2



############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2








##################### IMDN2 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn2_x2_epochs400_lr0001_vgpu40.out --job-name i2_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0001 --save imdn2_x2_epochs400_lr0001_vgpu40 --model IMDN2 --decay 100


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn2_x2_epochs400_lr0001_p100_b16_t8.out --job-name i2_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.001 --save imdn2_x2_epochs400_lr0001_p100_b16_t8 --model IMDN2 --decay 10 --n_threads 8 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn2_x2_epochs400_lr0001_p100_d10.out --job-name i2_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.001 --save imdn2_x2_epochs400_lr0001_p100_d10 --model IMDN2 --decay 10 --n_threads 8 --n_GPUs 2



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn2_x2_epochs400_lr00005_vgpu40_b25_100.out --job-name i2_2_5_40 tr.sh --scale 2 --epochs 200 --lr 0.0005 --save imdn2_x2_epochs400_lr0001_vgpu40_b25_100 --model IMDN2 --decay 25-100


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn2_x2_epoch200_lr00002_p100_d80_160.out --job-name i2_2_80160 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn2_x2_epoch200_lr00002_p100_d80_160 --model IMDN2 --decay 80-160 --n_threads 6 --n_GPUs 2







##################### IMDN3 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn3_x2_epoch200_lr00002_p100_d100.out --job-name i3_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn3_x2_epoch200_lr00002_p100_d100 --model IMDN3 --decay 100 --n_threads 6 --n_GPUs 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn3_x2_epoch200_lr00002_vgpu40_d100.out --job-name i3_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn3_x2_epoch200_lr00002_vgpu40_d100 --model IMDN3 --decay 100


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn3_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i3_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn3_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN3 --decay 80-160








##################### IMDN4 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn4_x2_epoch200_lr00002_p100_d80_160.out --job-name i4_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn4_x2_epoch200_lr00002_p100_d80_160 --model IMDN4 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn4_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i4_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn4_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN4 --decay 80-160








##################### IMDN5 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn5_x2_epoch200_lr00002_p100_d80_160.out --job-name i5_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn5_x2_epoch200_lr00002_p100_d80_160 --model IMDN5 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn5_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i5_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn5_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN5 --decay 80-160








##################### IMDN6 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn6_x2_epoch200_lr00002_p100_d80_160.out --job-name i6_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn6_x2_epoch200_lr00002_p100_d80_160 --model IMDN6 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn5_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i5_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn5_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN5 --decay 80-160










##################### IMDN7 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn7_x2_epoch200_lr00002_p100_d80_160.out --job-name i7_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn7_x2_epoch200_lr00002_p100_d80_160 --model IMDN7 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn7_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i7_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn7_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN7 --decay 80-160 --batch_size 8

############## test ################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn7_x2_epochs400_lr00002_p100.out --job-name i7_te te.sh --model IMDN7 --scale 2 --pre_train ../experiment/imdn7_x2_epoch200_lr00002_vgpu40_d80_160/model/model_best.pt --save test_imdn7_x2_epochs400_lr00002






##################### IMDN8 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn8_x2_epoch200_lr00002_p100_d80_160.out --job-name i8_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn8_x2_epoch200_lr00002_p100_d80_160 --model IMDN8 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn8_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i8_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn8_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN8 --decay 80-160 --batch_size 8









##################### IMDN9 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn9_x2_epoch200_lr00002_p100_d80_160.out --job-name i9_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn9_x2_epoch200_lr00002_p100_d80_160 --model IMDN9 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn9_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i9_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn9_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN9 --decay 80-160








##################### IMDN10 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn10_x2_epoch200_lr00002_p100_d80_160.out --job-name i10_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn10_x2_epoch200_lr00002_p100_d80_160 --model IMDN10 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn10_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i10_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn10_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN10 --decay 80-160








##################### IMDN11 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn11_x2_epoch200_lr00002_p100_d80_160.out --job-name i11_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn11_x2_epoch200_lr00002_p100_d80_160 --model IMDN11 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn11_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i11_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn11_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN11 --decay 80-160








##################### IMDN12 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn12_x2_epoch200_lr00002_p100_d80_160.out --job-name i12_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn12_x2_epoch200_lr00002_p100_d80_160 --model IMDN12 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn12_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i12_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn12_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN12 --decay 80-160







##################### IMDN14 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn14_x2_epoch200_lr00002_p100_d80_160.out --job-name i14_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn14_x2_epoch200_lr00002_p100_d80_160 --model IMDN14 --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn14_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name i14_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn14_x2_epoch200_lr00002_vgpu40_d80_160 --model IMDN14 --decay 80-160









##################### WRAN #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/wran_x2_epoch200_lr00002_p100_d80_160.out --job-name wran_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save wran_x2_epoch200_lr00002_p100_d80_160 --model WRAN --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/wran_x2_epoch200_lr00002_vgpu40_d80_160.out --job-name wran_2_40_80160 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save wran_x2_epoch200_lr00002_vgpu40_d80_160 --model WRAN --decay 80-160






##################### F0 #####################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/f0_x2_epochs200_lr00002_p100_d80_160.out --job-name f0_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save f0_x2_epochs200_lr00002_p100_d80_160 --model FSRCNN --decay 80-160 --n_threads 6 --n_GPUs 2




##################### F1 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/f1_x2_epochs200_lr00002_p100_d80_160.out --job-name f1_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save f1_x2_epochs200_lr00002_p100_d80_160 --model F1 --decay 80-160 --n_threads 6 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/f1_x2_epochs200_lr00002_v10.out --job-name f1_10 tr10.sh --scale 2 --epochs 200 --lr 0.0002 --save f1_x2_epochs200_lr00002_v10 --model F1 --decay 80-160


############### wave ###############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/f1_x2_epochs200_lr00002_p100_d80_160_wave.out --job-name f1_100w trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save f1_x2_epochs200_lr00002_p100_d80_160_wave --model F1 --decay 80-160 --n_threads 6 --n_GPUs 2 --loss 1*WAVELET

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/f1_x2_epochs200_lr00002_v10_d80_160_wave.out --job-name f1_10w tr10.sh --scale 2 --epochs 200 --lr 0.0002 --save f1_x2_epochs200_lr00002_v10_d80_160_wave --model F1 --decay 80-160 --loss 1*WAVELET --n_threads 4 --n_GPUs 1









##################### F1 #####################
############### wave ###############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/f2_x2_epochs200_lr00002_p100_d150_wave.out --job-name f2_100w trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save f2_x2_epochs200_lr00002_p100_d150_wave --model F2 --decay 150 --n_threads 8 --n_GPUs 2 --loss 1*W2











##################### IMDMKET #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x2_epoch200_lr00002_p100_d80_160.out --job-name iet_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdmket_x2_epoch200_lr00002_p100_d80_160 --model IMDMKET --decay 80-160 --n_threads 6 --n_GPUs 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x2_epoch200_lr00002.out --job-name iet_2_40 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdmket_x2_epoch200_lr00002 --model IMDMKET --decay 80-160 --batch_size 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x4_epoch200_lr00002.out --job-name iet_4_40 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdmket_x4_epoch200_lr00002 --model IMDMKET --decay 80-160 --batch_size 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x8_epoch200_lr00002.out --job-name iet_8_40 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket_x8_epoch200_lr00002 --model IMDMKET --decay 80-160 --batch_size 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x2_lr00002_g75_b2.out --job-name iet2_2_75 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdmket_x2_lr00002_g75_b2 --model IMDMKET --batch_size 2 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x4_lr00002_g75_b2.out --job-name iet4_2_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdmket_x4_lr00002_g75_b2 --model IMDMKET --batch_size 2 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x8_lr00002_g75_b2.out --job-name iet8_2_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket_x8_lr00002_g75_b2 --model IMDMKET --batch_size 2 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75






sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x2_lr00002_g75_b4.out --job-name iet2_4_75 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdmket_x2_lr00002_g75_b4 --model IMDMKET --batch_size 4 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x4_lr00002_g75_b4.out --job-name iet4_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdmket_x4_lr00002_g75_b4 --model IMDMKET --batch_size 4 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x8_lr00002_g75_b4.out --job-name iet8_4_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket_x8_lr00002_g75_b4 --model IMDMKET --batch_size 4 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket2_x8_25e5.out --job-name it8_4_25 tr.sh --scale 8 --epochs 200 --lr 0.000025 --save imdmket2_x8_25e5 --model IMDMKET2 --batch_size 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket3_x8_lr00002_g75.out --job-name i3_8_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket3_x8_lr00002_g75 --model IMDMKET3 --batch_size 2 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket4_x8_lr00002_g75.out --job-name i4_8_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket4_x8_lr00002_g75 --model IMDMKET4 --batch_size 2 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --loss 0.5*MSE+0.5*L1


############## test ################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket_x2.out --job-name i2_te te.sh --model IMDMKET --scale 2 --pre_train ../experiment/imdmket_x2_epoch200_lr00002/model/model_best.pt --save test_imdmket_x2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket_x4.out --job-name i4_te te.sh --model IMDMKET --scale 4 --pre_train ../experiment/imdmket_x4_epoch200_lr00002/model/model_best.pt --save test_imdmket_x4


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket_x8.out --job-name i8_te te.sh --model IMDMKET --scale 8 --pre_train ../experiment/imdmket_x8_epoch200_lr00002/model/model_best.pt --save test_imdmket_x8



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket_x2_p100.out --job-name i2_te100 test.sh --model IMDMKET --scale 2 --pre_train ../experiment/imdmket_x2_epoch200_lr00002/model/model_best.pt --save test_imdmket_x2_p100


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket_x4_p100.out --job-name i4_te100 test.sh --model IMDMKET --scale 4 --pre_train ../experiment/imdmket_x4_epoch200_lr00002/model/model_best.pt --save test_imdmket_x4_p100


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket_x8_p100.out --job-name i8_te100 test.sh --model IMDMKET --scale 8 --pre_train ../experiment/imdmket_x8_epoch200_lr00002/model/model_best.pt --save test_imdmket_x8_p100


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket_x8_p100_last.out --job-name i8l_te100 test.sh --model IMDMKET --scale 8 --pre_train ../experiment/imdmket_x8_epoch200_lr00002/model/model_latest.pt --save test_imdmket_x8_p100_last



--decay 20-40-60-80-100-120-140-160-180 --gamma 0.75











##################### IMDMKET2 #####################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket2_x8_lr00002_g75_b2.out --job-name it28_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket2_x8_lr00002_g75_b2 --model IMDMKET2 --batch_size 2 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75









##################### IMDN13 #####################
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn13_x2_epoch200_lr00002_p100_d80_160.out --job-name i12_2_100 trainp100.sh --scale 2 --epochs 200 --lr 0.0002 --save imdn12_x2_epoch200_lr00002_p100_d80_160 --model IMDN12 --decay 80-160 --n_threads 6 --n_GPUs 2






sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn13_x8_ep200_lr00002_g75.out --job-name i13_8_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdn13_x8_ep200_lr00002_g75 --model IMDN13 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn13_x8_ep200_lr00002_g75_100.out --job-name i13_8_75100 trainp100.sh --scale 8 --epochs 200 --lr 0.0002 --save imdn13_x8_ep200_lr00002_g75_100 --model IMDN13 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --n_threads 8 --n_GPUs 2




















################X8################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/fsrcnn_x8_epochs400_lr00001_p100.out --job-name f8_100 trainp100.sh --scale 8 --epochs 400 --lr 0.0001 --save fsrcnn_x8_epochs400_lr00001_p100 --model FSRCNN --pre_train ../experiment/fsrcnn_x4_epochs400_lr00001/model/model_best.pt --loss 1*MSE

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/fsrcnn_x8_epochs400_lr0001_p100.out --job-name f8_100 trainp100.sh --scale 8 --epochs 400 --lr 0.001 --save fsrcnn_x8_epochs400_lr0001_p100 --model FSRCNN --pre_train ../experiment/fsrcnn_x4_epochs400_lr00001/model/model_best.pt --loss 1*MSE



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/edsr_x8_epochs300_lr000005_base.out --job-name edsrtr8 trainp100.sh --scale 8 --epochs 300 --lr 0.00005 --save edsr_x8_epochs300_lr000005_base --model EDSR --pre_train ../experiment/edsr_x4_epochs300_lr00001/model/model_best.pt

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/edsr_x8_epochs300_lr000001_base.out --job-name edsrtr8 trainp100.sh --scale 8 --epochs 300 --lr 0.00001 --save edsr_x8_epochs300_lr000001_base --model EDSR --pre_train ../experiment/edsr_x4_epochs300_lr00001/model/model_best.pt




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x8_epochs400_lr00001_p100_bk4_b16_nop.out --job-name ebrn8_100_nop trainp100.sh --scale 8 --epochs 400 --lr 0.0001 --save ebrn_x8_epochs400_lr00001_p100_bk4_b16_nop --model EBRN --n_threads 4 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/ebrn_x8_epochs400_lr00001_vgpu40_bk4_b16_nop.out --job-name ebrn8_40_nop tr.sh --scale 8 --epochs 400 --lr 0.0001 --save ebrn_x8_epochs400_lr00001_vgpu40_nop --model EBRN

################################################


####################Testing####################

################vgpu40################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_edsr_x2.out --job-name edsrte te.sh --model EDSR --scale 2 --pre_train ../experiment/edsr_baseline_x2_epoch300_lr/model/model_best.pt

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_fsrcnn_x2_epochs400_lr00001.out --job-name fffte te.sh --model FSRCNN --scale 2 --pre_train ../experiment/fsrcnn_x2_epochs400_lr00001/model/model_best.pt

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_edsr_x4.out --job-name edsrte4 test.sh --model EDSR --scale 4 --pre_train ../experiment/edsr_x4_epochs300_lr00001/model/model_best.pt --save test_edsr_x4

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_rdn_x2_epochs200_lr00001.out --job-name rdnte2 te.sh --model RDN --scale 2 --pre_train /home/Student/s4427443/edsr40/src/rdn2_model_best.pt --save test_rdn_x2_epochs200_lr00001

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_rdn_x2_epochs200_lr000001.out --job-name rdnte2 test.sh --model RDN --scale 2 --batch_size 4 --pre_train ../experiment/rdn_x2_epochs200_lr000001/model/model_best.pt --save test_rdn_x2_epochs200_lr000001 --n_threads 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_rdn_x4_epochs200_lr0000001.out --job-name rdnte4 test.sh --model RDN --scale 4 --batch_size 4 --pre_train ../experiment/rdn_x4_epochs200_lr0000001/model/model_best.pt --save test_rdn_x4_epochs200_lr0000001 --n_threads 2


################p100################

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_edsr_x2.out --job-name edsrte2 test.sh --model EDSR --scale 2 --pre_train ../experiment/edsr_baseline_x2_epoch300_lr/model/model_best.pt --save test_edsr_x2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_fsrcnn_x2_epochs400_lr00001.out --job-name fffte2 test.sh --model FSRCNN --scale 2 --pre_train ../experiment/fsrcnn_x2_epochs400_lr00001/model/model_best.pt --save test_fsrcnn_x2_epochs400_lr00001

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_fsrcnn_x4_epochs400_lr00001.out --job-name fffte4 test.sh --model FSRCNN --scale 4 --pre_train ../experiment/fsrcnn_x4_epochs400_lr00001/model/model_best.pt --save test_fsrcnn_x4_epochs400_lr00001



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_fsrcnn_x2_epochs400_lr00001.out --job-name fffte2 test.sh --model FSRCNN --scale 2 --pre_train ../experiment/fsrcnn_x2_epochs400_lr00001/model/model_best.pt --save test_fsrcnn_x2_epochs400_lr00001



################## FSRCNN ##################
############ X4 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_fsrcnn_x4_epochs400_lr00001.out --job-name fffte4 test.sh --model FSRCNN --scale 4 --pre_train ../experiment/fsrcnn_x4_epochs400_lr00001/model/model_best.pt --save test_fsrcnn_x4_epochs400_lr00001

############ X8 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_fsrcnn_x8_epochs400_lr00001.out --job-name fte8 te.sh --model FSRCNN --scale 8 --pre_train ../experiment/fsrcnn_x8_epochs400_lr00001_p100/model/model_best.pt --save test_fsrcnn_x8_epochs400_lr00001 --n_threads 4





################## EDSR ##################
############ X2 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_edsr_x2.out --job-name edsrte2 te.sh --model EDSR --scale 2 --pre_train ../experiment/edsr_baseline_x2_epoch300_lr/model/model_best.pt --save test_edsr_x2


############ X4 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_edsr_x4.out --job-name edsrte4 te.sh --model EDSR --scale 4 --pre_train ../experiment/edsr_x4_epochs300_lr00001/model/model_best.pt --save test_edsr_x4


############ X8 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_edsr_x8.out --job-name edsrte8 te.sh --model EDSR --scale 8 --pre_train ../experiment/edsr_x8_epochs300_lr000001_base/model/model_best.pt --save test_edsr_x8

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_edsr_x8_last.out --job-name edsrte8 te.sh --model EDSR --scale 8 --pre_train ../experiment/edsr_x8_epochs300_lr000001_base/model/model_latest.pt --save test_edsr_x8_last






################## EBRN ##################
############ X2 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_ebrn_x2.out --job-name ebrnte2 te.sh --model EBRN --scale 2 --pre_train ../experiment/ebrn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_ebrn_x2 --n_threads 4


############ X4 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_ebrn_x4.out --job-name ebrnte4 te.sh --model EBRN --scale 4 --pre_train ../experiment/ebrn_x4_epochs400_lr00001_vgpu40_nop/model/model_best.pt --save test_ebrn_x4 --n_threads 4


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_ebrn_x4_100.out --job-name ebrnte4 test.sh --model EBRN --scale 4 --pre_train ../experiment/ebrn_x4_epochs400_lr00001_vgpu40_nop/model/model_best.pt --save test_ebrn_x4_100 --n_threads 4


############ X8 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_ebrn_x8.out --job-name ebrnte8 te.sh --model EBRN --scale 8 --pre_train ../experiment/ebrn_x8_epochs400_lr00001_vgpu40_nop/model/model_best.pt --save test_ebrn_x8 --n_threads 8


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_ebrn_x8_100.out --job-name ebrnte8 test.sh --model EBRN --scale 8 --pre_train ../experiment/ebrn_x8_epochs400_lr00001_vgpu40_nop/model/model_best.pt --save test_ebrn_x8_100 --n_threads 4




################## IMDN ##################
############ X2 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_epochs400_lr00001_p100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn_x2_epochs400_lr00001_vgpu40/model/model_best.pt --save test_imdn_x2_epochs400_lr00001_p100 --n_threads 8 --n_GPUs 2

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x2_100.out --job-name i_te2 test.sh --model IMDN --scale 2 --pre_train ../experiment/imdn1_x2_epoch200_lr00002_p100_d80_160/model/model_best.pt --save test_imdn_x2_100 --n_threads 8 --n_GPUs 2


############ X4 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x4_100.out --job-name i_te4 test.sh --model IMDN --scale 4 --pre_train ../experiment/imdn1_x4_epoch200_lr00002_p100_d80_160/model/model_best.pt --save test_imdn_x4_100 --n_threads 8 --n_GPUs 2


############ X8 ############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn_x8_100.out --job-name i_te8 test.sh --model IMDN --scale 8 --pre_train ../experiment/imdn1_x8_epoch200_lr00002_p100_d80_160/model/model_best.pt --save test_imdn_x8_100 --n_threads 8 --n_GPUs 2




################################################

python3 main.py $* \
				--patch_size 256 \
                --dir_data /home/Student/s4427443/edsr1/MRI \
                --data_range 1-10/31678-35197 \
                --test_only \
                --test_every 1 \
                --batch_size 16 \
                --n_colors 1 \
                --reset \
                --save_results
				

png generation

sbatch png.sh --scale 4

sbatch png.sh --scale 8

#SBATCH --partition=vgpu
#SBATCH --gres=gpu:1

# number of CPU cores per task
#SBATCH --cpus-per-task=8

tar -czvf fsrcnn_test_results.tgz fsrcnn/output/x2/test_results/

sbatch -o /home/Student/s4427443/edsr40/sbStdout/edsr2ssim.out --job-name edsr2ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x2/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/edsr4ssim.out --job-name edsr4ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x4/results-DIV2K --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/fsrcnn2ssim.out --job-name fsrcnn2ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_fsrcnn_x2_epochs400_lr00001/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/fsrcnn4ssim.out --job-name fsrcnn4ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_fsrcnn_x4_epochs400_lr00001/results-DIV2K --scale 4


sbatch -o /home/Student/s4427443/edsr40/sbStdout/edsr2ssim.out --job-name edsr2ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x2/results-DIV2K && sbatch -o /home/Student/s4427443/edsr40/sbStdout/edsr4ssim.out --job-name edsr4ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x4/results-DIV2K && sbatch -o /home/Student/s4427443/edsr40/sbStdout/fsrcnn2ssim.out --job-name fsrcnn2ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_fsrcnn_x2_epochs400_lr00001/results-DIV2K && sbatch -o /home/Student/s4427443/edsr40/sbStdout/fsrcnn4ssim.out --job-name fsrcnn4ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_fsrcnn_x4_epochs400_lr00001/results-DIV2K



############## SSIM ##############
##### edsr #####
sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_edsr2.out --job-name ssim_edsr2 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x2/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_edsr4.out --job-name ssim_edsr4 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x4/results-DIV2K --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_edsr8.out --job-name ssim_edsr8 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x8/results-DIV2K --scale 8

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_edsr8_last.out --job-name ssim_edsr8_last h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x8_last/results-DIV2K --scale 8 --batch_size 8



##### fsrcnn #####
sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_fsrcnn2.out --job-name ssim_fsrcnn2 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_fsrcnn_x2_epochs400_lr00001/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_fsrcnn4.out --job-name ssim_fsrcnn4 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_fsrcnn_x4_epochs400_lr00001/results-DIV2K --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_fsrcnn8.out --job-name ssim_fsrcnn8 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_fsrcnn_x8_epochs400_lr00001/results-DIV2K --scale 8


##### bicubic #####
sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_bic2.out --job-name ssim_bic2 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_bicubic_x2 --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_bic4.out --job-name ssim_bic4 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_bicubic_x4 --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_bic8.out --job-name ssim_bic8 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_bicubic_x8 --scale 8


##### ebrn #####
sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_ebrn2.out --job-name ssim_ebrn2 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_ebrn_x2/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_ebrn4.out --job-name ssim_ebrn4 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_ebrn_x4/results-DIV2K --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_ebrn8.out --job-name ssim_ebrn8 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_ebrn_x8_100/results-DIV2K --scale 8


##### imdn #####
sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdn2.out --job-name ssim_imdn2 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdn_x2_100/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdn4.out --job-name ssim_imdn4 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdn_x4_100/results-DIV2K --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdn8.out --job-name ssim_imdn8 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdn_x8_100/results-DIV2K --scale 8


##### imdmket #####
sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdmket2.out --job-name ssim_imdmket2 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdmket_x2_p100/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdmket4.out --job-name ssim_imdmket4 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdmket_x4_p100/results-DIV2K --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdmket8.out --job-name ssim_imdmket8 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdmket_x8_p100/results-DIV2K --scale 8

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdmket8_last.out --job-name ssim_imdmket8_last h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdmket_x8_p100_last/results-DIV2K --scale 8


tail -2 ssim_edsr2.out
tail -2 ssim_edsr4.out
tail -2 ssim_edsr8.out

tail -2 ssim_ebrn2.out
tail -2 ssim_ebrn4.out
tail -2 ssim_ebrn8.out

tail -2 ssim_imdn2.out
tail -2 ssim_imdn4.out
tail -2 ssim_imdn8.out

tail ssim_imdmket2.out
tail ssim_imdmket4.out
tail ssim_imdmket8.out

tail -2 ssim_imdmket2.out
tail -2 ssim_imdmket4.out
tail -2 ssim_imdmket8.out
tail -2 ssim_imdmket8_last.out





1-10/31678-35197
[r.split('-') for r in "1-10/31678-35197".split('/')]
data_range = ['31678', '35197']
begin, end = list(map(lambda x: int(x), data_range))

sbatch -o /home/Student/s4427443/edsr40/sbStdout/bic.out --job-name bic bi.sh

sbatch -o /home/Student/s4427443/edsr40/sbStdout/bic.out --job-name bic bi.sh --x8


sbatch -o /home/Student/s4427443/edsr40/sbStdout/bic2ssim.out --job-name bic2ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_bicubic_x2 --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/bic4ssim.out --job-name bic4ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_bicubic_x4 --scale 4


sbatch -o /home/Student/s4427443/edsr40/sbStdout/rdn2ssim.out --job-name rdn2ssim h.sh --sr /home/Student/s4427443/edsr40/experiment/test_rdn_x2_epochs200_lr00001/results-DIV2K --scale 2



tar -czvf DIV2K.tgz /home/Student/s4427443/edsr40/MRI/DIV2K


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_edsr_x8_last.out --job-name edsrte8 test.sh --model EDSR --scale 8 --pre_train ../experiment/edsr_x8_epochs300_lr000001_base/model/model_latest.pt --save test_edsr_x8_last --batch_size 8

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_edsr8_last.out --job-name ssim_edsr8_last h.sh --sr /home/Student/s4427443/edsr40/experiment/test_edsr_x8_last/results-DIV2K --scale 8

tail -2 sbStdout/ssim_edsr8_last

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/edsr_x8_lr00002_g75.out --job-name edsr8_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save edsr_x8_lr00002_g75 --model EDSR --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 8


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket3_x8_lr00002_g75.out --job-name i3_8_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket3_x8_lr00002_g75 --model IMDMKET3 --batch_size 2 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75






sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x4_2040.out --job-name it4_2040 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdmket_x4_2040 --model IMDMKET --batch_size 2 --decay 20-40 --gamma 0.4

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket_x8_20.out --job-name it8_20 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket_x8_20 --model IMDMKET --batch_size 2 --decay 20 --gamma 0.125


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket3_x8_lr00002_g75.out --job-name i3_8_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket3_x8_lr00002_g75 --model IMDMKET3 --batch_size 2 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75



############### imdn20 + head + tail + cca ###############                  27.728 (Best: 27.745 @epoch 114)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn20_x4_lr00002_g75.out --job-name i20_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn20_x4_lr00002_g75 --model IMDN20 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --n_threads 8



############### imdn21 + head + tail + cca + mas ###############
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn21_x4_lr00002_g75.out --job-name i21_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn21_x4_lr00002_g75 --model IMDN21 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --n_threads 8



############### imdn22 + head5 + tail + cca ###############                27.728 (Best: 27.745 @epoch 114)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn22_x4_lr00002_g75.out --job-name i22_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn22_x4_lr00002_g75 --model IMDN22 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --n_threads 8



############### imdn23 + head + tail + cca + csa(correct) ###############  27.704 (Best: 27.717 @epoch 121) ----v
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn23_x4_lr00002_g75.out --job-name i23_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn23_x4_lr00002_g75 --model IMDN23 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75



############### imdn24 + head + tail + cca(x*x) + csa ###############      27.691 (Best: 27.739 @epoch 76)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn24_x4_lr00002_g75.out --job-name i24_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn24_x4_lr00002_g75 --model IMDN24 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --n_threads 6



############### imdn25 + head + tail + cca + csa + mkam ###############    27.715 (Best: 27.746 @epoch 76)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn25_x4_lr00002_g75.out --job-name i25_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn25_x4_lr00002_g75 --model IMDN25 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75



############### imdn26 + head + tail + cca + csa + mkm ###############    27.724 (Best: 27.763 @epoch 101)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn26_x4_lr00002_g75.out --job-name i26_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn26_x4_lr00002_g75 --model IMDN26 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75



############### imdn27 + head + tail + cca(+1) + csa(+1) ###############  27.717 (Best: 27.755 @epoch 92) ----v
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn27_x4_lr00002_g75.out --job-name i27_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn27_x4_lr00002_g75 --model IMDN27 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75



############### imdn28 + head + tail + cca(+1) ###############            27.725 (Best: 27.756 @epoch 94)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn28_x4_lr00002_g75.out --job-name i28_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn28_x4_lr00002_g75 --model IMDN28 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75



############### imdn29 + head + tail + cca(+1) + csa(+1) + mkm ###############   27.700 (Best: 27.769 @epoch 47) ----v
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn29_x4_lr00002_g75.out --job-name i29_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn29_x4_lr00002_g75 --model IMDN29 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75



############### imdn30 + head + tail + cca(+1) + csa(+1) + mkm + dense ###############    27.731 (Best: 27.802 @epoch 60) ----v
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn30_x4_lr00002_g75.out --job-name i30_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn30_x4_lr00002_g75 --model IMDN30 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn30_x4_lr00002_g75_100.out --job-name imdn30test test.sh --model IMDN30 --scale 4 --pre_train ../experiment/imdn30_x4_lr00002_g75/model/model_best.pt --save test_imdn30_x4_lr00002_g75_100 --n_threads 8 --n_GPUs 2



############### imdn31 + head + tail ###############              27.687 (Best: 27.710 @epoch 110)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn31_x4_lr00002_g75.out --job-name i31_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn31_x4_lr00002_g75 --model IMDN31 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdn31_x4_lr00002_g75_100.out --job-name imdn31test test.sh --model IMDN31 --scale 4 --pre_train ../experiment/imdn31_x4_lr00002_g75/model/model_best.pt --save test_imdn31_x4_lr00002_g75_100 --n_threads 8 --n_GPUs 2



############### imdn32 + head + tail + ca + sa ###############            27.721 (Best: 27.738 @epoch 103)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdn32_x4_lr00002_g75.out --job-name i32_4_75 tr20.sh --scale 4 --epochs 200 --lr 0.0002 --save imdn32_x4_lr00002_g75 --model IMDN32 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --n_threads 4




############### imdmket5 + head + tail + cca(+1) + csa(+1) + mkm + dense + Transformer ###############    
####### x4 27.757 #######            27.757 (Best: 27.928 @epoch 34)
sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket5_x4_lr00002_g75.out --job-name i5_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdmket5_x4_lr00002_g75 --model IMDMKET5 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 4

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket5_x4_lr00002_g75_100.out --job-name i5test test.sh --model IMDMKET5 --scale 4 --pre_train ../experiment/imdmket5_x4_lr00002_g75/model/model_best.pt --save test_imdmket5_x4_lr00002_g75_100 --n_threads 8 --n_GPUs 2



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket5_x2_lr00002_g75.out --job-name i5_2_75 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdmket5_x2_lr00002_g75 --model IMDMKET5 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 4



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket5_x8_lr00002_g75.out --job-name i5_8_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket5_x8_lr00002_g75 --model IMDMKET5 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 4







###### SSIM ######      31 30 29 27 23 
sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdn.out --job-name ssim_imdn h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdn_x4_lr00002_g75_100/results-DIV2K --scale 2




sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket6_x2_lr00002_g75.out --job-name i6_2_75 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdmket6_x2_lr00002_g75 --model IMDMKET6 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 4





sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket6_x4_lr00002_g75.out --job-name i6test te.sh --model IMDMKET6 --scale 4 --pre_train ../experiment/imdmket6_x4_lr00002_g75/model/model_best.pt --save test_imdmket6_x4_lr00002_g75 --n_threads 8 --n_GPUs 2



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket6_x4_lr00002_g75_latest.out --job-name i6test te.sh --model IMDMKET6 --scale 4 --pre_train ../experiment/imdmket6_x4_lr00002_g75/model/model_latest.pt --save test_imdmket6_x4_lr00002_g75_latest --n_threads 8 --n_GPUs 2







sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket6_x4_lr00002_g75.out --job-name i6test test.sh --model IMDMKET6 --scale 4 --pre_train ../experiment/imdmket6_x4_lr00002_g75/model/model_best.pt --save test_imdmket6_x4_lr00002_g75 --n_threads 2



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket6_x4_lr00002_g75_latest_100.out --job-name i6test test.sh --model IMDMKET6 --scale 4 --pre_train ../experiment/imdmket6_x4_lr00002_g75/model/model_latest.pt --save test_imdmket6_x4_lr00002_g75_latest --n_threads 2




################### imdmket7 

sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket7_x4_lr00002_g75.out --job-name i7_4_75 tr.sh --scale 4 --epochs 200 --lr 0.0002 --save imdmket7_x4_lr00002_g75 --model IMDMKET7 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 2



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket7_x2_lr00002_g75.out --job-name i7_2_75 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdmket5_x2_lr00002_g75 --model IMDMKET7 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 2



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket7_x8_lr00002_g75.out --job-name i7_8_75 tr.sh --scale 8 --epochs 200 --lr 0.0002 --save imdmket5_x8_lr00002_g75 --model IMDMKET7 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 2



sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/imdmket7_x2_lr00002_g75_continue.out --job-name i7_2_75 tr.sh --scale 2 --epochs 200 --lr 0.0002 --save imdmket7_x2_lr00002_g75 --model IMDMKET7 --decay 20-40-60-80-100-120-140-160-180 --gamma 0.75 --batch_size 2 --resume 187





sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket7_x2_lr00002_g75_100.out --job-name i7test2 test.sh --model IMDMKET7 --scale 2 --pre_train ../experiment/imdmket7_x2_lr00002_g75/model/model_best.pt --save test_imdmket7_x2_lr00002_g75_100 --n_threads 8 --n_GPUs 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket7_x4_lr00002_g75_100.out --job-name i7test4 test.sh --model IMDMKET7 --scale 4 --pre_train ../experiment/imdmket7_x4_lr00002_g75/model/model_best.pt --save test_imdmket7_x4_lr00002_g75_100 --n_threads 8 --n_GPUs 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket7_x8_lr00002_g75_100.out --job-name i7test8 test.sh --model IMDMKET7 --scale 8 --pre_train ../experiment/imdmket7_x8_lr00002_g75/model/model_best.pt --save test_imdmket7_x8_lr00002_g75_100 --n_threads 8 --n_GPUs 2






sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket7_x4_lr00002_g75_100_latest.out --job-name i7test4l test.sh --model IMDMKET7 --scale 4 --pre_train ../experiment/imdmket7_x4_lr00002_g75/model/model_latest.pt --save test_imdmket7_x4_lr00002_g75_100_latest --n_threads 8 --n_GPUs 2


sbatch -o /home/Student/s4427443/edsr40/src/sbatchStdout/test_imdmket7_x8_lr00002_g75_100_latest.out --job-name i7test8l test.sh --model IMDMKET7 --scale 8 --pre_train ../experiment/imdmket7_x8_lr00002_g75/model/model_latest.pt --save test_imdmket7_x8_lr00002_g75_100_latest --n_threads 8 --n_GPUs 2






tail sbatchStdout/test_imdmket7_x2_lr00002_g75_100.out
tail sbatchStdout/test_imdmket7_x4_lr00002_g75_100.out
tail sbatchStdout/test_imdmket7_x8_lr00002_g75_100.out

tail sbatchStdout/test_imdmket7_x4_lr00002_g75_100.out
tail sbatchStdout/test_imdmket7_x4_lr00002_g75_100_latest.out
tail sbatchStdout/test_imdmket7_x8_lr00002_g75_100.out
tail sbatchStdout/test_imdmket7_x8_lr00002_g75_100_latest.out




sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdmket7_2.out --job-name ssim_imdmket7_2 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdmket7_x2_lr00002_g75_100/results-DIV2K --scale 2

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdmket7_4.out --job-name ssim_imdmket7_4 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdmket7_x4_lr00002_g75_100/results-DIV2K --scale 4

sbatch -o /home/Student/s4427443/edsr40/sbStdout/ssim_imdmket7_8.out --job-name ssim_imdmket7_8 h.sh --sr /home/Student/s4427443/edsr40/experiment/test_imdmket7_x8_lr00002_g75_100/results-DIV2K --scale 8

tail -2 sbStdout/ssim_imdmket7_2.out
tail -2 sbStdout/ssim_imdmket7_4.out
tail -2 sbStdout/ssim_imdmket7_8.out
