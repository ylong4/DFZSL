#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 原seed 9182

import os
os.system('''OMP_NUM_THREADS=4 python /data/ylong/dfgzsl2/tfvaegan/train_images.py \
--gammaD 10 --gammaG 10 --gzsl --encoded_noise --manualSeed 806 --cuda \
--image_embedding vlpt-simulator-ViTB16 \
--class_embedding vlpt-simulator-clip \
--preprocessing --nepoch 120 --syn_num 1800 --ngh 1024 --ndh 1024 --lambda1 10 --critic_iter 5 \
--nclass_all 50 --dataroot /data/ylong/datasets --dataset AWA2 \
--batch_size 64 --nz 512 --latent_size 512 --attSize 512 --resSize 512 \
--lr 0.00003 --classifier_lr 0.00005 --recons_weight 0.1 --freeze_dec \
--feed_lr 0.0007 --dec_lr 0.0005 --feedback_loop 2 --a1 0.01 --a2 0.01 --ratio 1.0''')
# --syn_num 1200 经验
# --nclass_all
# 绝对路径
# --classifier_lr 0.00001 就5个0
# --image_embedding  vlpt-simulator-ViTB16
# --class_embedding  vlpt-simulator-clip /coop
# -dataroot


# best :{'seen': tensor(0.9377), 'unseen': tensor(0.9329), 'h_acc': tensor(0.9353)}
# best can :{'seen/unseen': tensor(0.9353), 'lr': 3e-05, 'classifier_lr': 5e-05, 'feed_lr': 0.0007,
# 'dec_lr': 0.0005, 'ratio': 1.0, 'syn_num': 1800}