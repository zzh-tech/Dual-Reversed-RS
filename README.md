# Dual-Reversed-RS

[ECCV2022 Oral] Bringing Rolling Shutter Images Alive with Dual Reversed Distortion  
by Zhihang Zhong, Mingdeng Cao, Xiao Sun, Zhirong Wu, Zhongyi Zhou, Yinqiang Zheng, Stephen Lin, and Imari Sato

## Dataset

Please download the dataset from
this [link](https://drive.google.com/file/d/1DuJphkVpvsNjgPs73y_sm4WZ8tzfxOZf/view?usp=sharing).

## Train

Please replace 'save_dir' and 'data_dir' accordingly. 'data_dir' contains the unzipped dataset directory.

```shell
CUDA_VISIBLE_DEVICES=0 python main.py --frames 3 --save_dir=./experiments --data_root=/home/zhong/Dataset/ --dataset=RS-GOPRO_DS --model=DIFE_MulCatFusion
```

## Pretrained models

Please download the pretrained checkpoints from this [link](https://drive.google.com/drive/folders/19RNmG10KMCNRi3VjD2B72GC5MBMlwOit?usp=sharing).

## Inference

Please refer to [inference.py](inference.py)

## TODO

- [ ] Project website
- [ ] Reformat the code for better readability
- [ ] list dependencies

## Citation

If our code or RS-GOPRO dataset are useful for your research, please consider citing:

```bibtex
@article{zhong2022bringing,
  title={Bringing rolling shutter images alive with dual reversed distortion},
  author={Zhong, Zhihang and Cao, Mingdeng and Sun, Xiao and Wu, Zhirong and Zhou, Zhongyi and Zheng, Yinqiang and Lin, Stephen and Sato, Imari},
  journal={arXiv preprint arXiv:2203.06451},
  year={2022}
}
```