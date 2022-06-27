由于文件大小所限，没有data文件夹，助教评测时需要将data文件夹放到当前目录下
报告里提到的各个实验的具体结果和细节见 experiments 文件夹

train.py里包含了测试的过程，故不需要再执行evaluate.py

early fusion实验运行
python train.py --model early_fusion

late fusion最佳实验结果复现实验运行
python train.py --model late_fusion

late fusion最佳alpha调参运行：（需要安装wandb库）
wandb sweep wandb.yaml
再根据提示运行对应的sweep

LSTM 实验运行
python train_only.py

train.py 和 model.py中有详细的注释供助教检查


