# 多模态情感分析项目

## 项目概述
本项目实现了一个基于图像和文本的多模态情感分析系统，通过融合视觉和语言特征来提升情感分类的准确性。

## 项目结构
```text
project/
├──datasearch.ipynb                        # 文件编码类型代码（查询文件编码类型）
|──base.ipynb                              # 基线模型部分代码
|──iprove优化.ipynb                        # 多模态融合模型代码
|── picture/                               # 可视化图表
│   ├── learning_rate_tune_results.png     # 学习率调参图
│   ├── weight_decay_tune_resunlts.png     # 权重衰减调参图
│   ├── train_metrics.png                  # 基线模型最佳超参数训练图
│   ├── ablation_acc_compare.png           # 消融实验对比图
│   ├── model_training_results.csv         # 多模态融合模型训练数据
│   └── model_training_curves.png          # 多模态融合模型训练图
├── requirements.txt                       # 依赖包列表
|── README.md                              # 项目说明
└── test_prediction.txt                    # 测试集结果
```

## 依赖要求

Python环境：3.11

- torch==2.3.1+cu121
- torchvision==0.18.1+cu121
- transformers==4.37.2
- numpy==1.24.3
- scipy==1.10.1
- sklearn==1.18.0
- matplotlib==3.10.8

带 CUDA 后缀的 PyTorch 无法直接通过普通 pip 安装，单独用官方命令安装：
```python
pip3 install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
再运行文本 requirements.txt
```python
pip install -r requirements.txt
```

## 代码完整流程

1. 解压实验所使用的数据project5.zip

2. 运行datasearch.ipynb的两个代码块，查看数据集的文本编码类型
```python
jupyter nbconvert --execute --to notebook --stdout datasearch.ipynb
```
3. 依次运行base.ipynb文件中的所有代码块，会分别进行“学习率、权重衰减调参”、“最优超参数训练”、“消融实验对比”实验
```python
jupyter nbconvert --execute --to notebook --stdout base.ipynb
```

4. 同样依次运行improve优化.ipynb文件中的所有代码块，会进行“多模态融合模型训练对比”、“数据增强对比”实验
```python
jupyter nbconvert --execute --to notebook --stdout improve优化.ipynb
```

## 参考

实验的进行参考了部分GitHub仓库：

- [CLIP、BLIP](https://github.com/pramanik-souvik/comparing-vlms-for-movie-genre-prediction/tree/main)

- [早期融合、后期融合](https://github.com/pramanik-souvik/comparing-vlms-for-movie-genre-prediction/tree/main)