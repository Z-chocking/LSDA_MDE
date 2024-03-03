# 基于长短距离注意力的单目深度估计网络

## 环境配置

请参考NewCRFs



## 数据集

请参考BTS.



## 训练

对于KITTI：

```
python lsda/train.py configs/arguments_train_kittieigen.txt
```



对于NYU：

```
python lsda/train.py configs/arguments_train_nyu.txt
```



## 评估

NYU:

```
CUDA_VISIBLE_DEVICES=0 python lsda/eval.py configs/arguments_eval_nyu.txt
```

KITTI:

```
CUDA_VISIBLE_DEVICES=0 python newcrfs/eval.py configs/arguments_eval_kittieigen.txt 
```

