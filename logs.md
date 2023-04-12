## base

```
python3 main.py somethingv2 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb

python3 test_models.py somethingv2 --weights=checkpoint/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar --test_segments=8 --batch_size=72 -j 24 --test_crops=3 --twice_sample
```

Class Accuracy 54.83%
Overall Prec@1 61.68% Prec@5 87.38%

## base-minus

```
python3 main.py somethingv2 RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 --lr_steps 20 40 --epochs 50 --batch-size 64 -j 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 --shift --shift_div=8 --shift_place=blockres --npb --exp=minus

python3 test_models.py somethingv2 --weights=checkpoint/TSM_somethingv2_RGB_resnet50_shift8_blockres_avg_segment8_e50/ckpt.best.pth.tar --test_segments=8 --batch_size=72 -j 24 --test_crops=3 --twice_sample --exp=minus
```

Class Accuracy 55.34%
Overall Prec@1 62.13% Prec@5 87.29%
