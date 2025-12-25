# Multimodal-fNIRS-EEG-Dataset
Multimodal fNIRS-EEG Dataset for Unilateral Limb Motor Imagery
- - 原始 EEG 读取与预处理脚本（MNE-Python）
  - 原始 fNIRS `.TXT`/`.csv` 解析、手动打标时间读取、带通滤波与 Epoch 构建脚本
  - EEG / fNIRS 示例可视化
建议用户在 Python 3.10+、`mne`、`numpy`、`scipy`、`pandas`、`matplotlib` 环境下运行。

Code\Code\Preprocessing\下为EEG和fNIRS预处理代码
其中EEG_process.py、EEGPreprocess.m为EEG预处理代码
2024_11_11_snirf_trans_merge.ipynb和no_mrk.ipynb分别为fNIRS自动打标和后期手动补标签的的预处理代码
Code\Plot\下为分析绘图代码
Code\Classification下为分类代码






