# C2F-CSLRNet

本仓库整理了C2F-CSLRNet的研究组件、科学配置、修正后的实验结果和小型冻结评分器权重。
论文作者为Siyu Zhu、Jiahao Zhang、Jie Liu、Yonghao Yang、Yanshan Qian、Yuli Chen、Lin Miao和Xiulei Liu。

当前论文的数据入口为[公开研究数据1.0.0](https://github.com/yinnierzaidemeng/C2F-CSLRNet/releases/tag/data-v1.0.0)。
下载`C2F_public_research_data_20261018.zip`，从`PUBLIC_DATA_INDEX.json`读取汇总、逐文件结果、配置及来源记录。
公开副本对执行路径作了规范化，结果数值与原记录保持一致。许可与引用入口见[data/README.md](data/README.md)。

## 从哪里开始

- 阅读英文 [README](README.md) 了解结构。
- 阅读 [实现说明](docs/IMPLEMENTATION.md) 了解模型、修正标准化及评分器的输入输出。
- `results/`保留10月3日的历史快照；当前论文采用上方版本化数据发布中的汇总和逐文件结果。
- `results/retrieval/`另含合成关系恢复和固定候选权重干预的现成记录。
- `checkpoints/` 只提供小型评分器权重，不是全部预测器模型。

无需安装科研依赖即可执行文件核验：

```text
python scripts/verify_release.py
python scripts/verify_artifacts.py
python scripts/verify_public_data.py /path/to/C2F_public_research_data_20261018.zip
```

检查只读取文件并核对哈希，不训练、不推理、不计算指标。

## 两组结果不能混用

保留预测器评分组中，C2F动态VUS-PR为0.4245；统一训练结构组中，完整C2F为0.4153。
它们来自不同预测器拟合，按各自结果入口分别读取。当前论文采用文件等权评价，
完整六指标、种子区分、配对估计及不利比较保留于版本化数据发布中。

本次整理没有进行新的实验，也没有验证从原始数据开始的一键端到端复现。
原始基准、完整预测器检查点和完整残差轨迹不在此仓库中；具体范围见
[复现范围说明](docs/REPRODUCIBILITY.md)。

原创代码及相关说明采用MIT；原始研究结果表、科学配置和核验记录采用CC BY 4.0。
评分器权重和第三方材料保留原许可范围，具体见
[许可说明](docs/LICENSING.md)。
