Sorting numbers with pointer networks(Ptr-Net)
 
# 问题描述
    利用利用 pointer network 对数字进行排序

 

# 数据集
    在100 以内的整合集 中随机采样 5 个 （或者10个）数字 用来排序，对应的 y 是这个 五个数字从小到大排序完成后的 地址。如：

    x =[3,97,42,26,30] ，y=[0,3,4,2,1]。

    在 0~1 之间采样 5 个浮点数，排序。

    x = [0.54431329, 0.64373097, 0.9927333 , 0.70941862, 0.10016056] , y = [4,0,1,3,2]

# 题目要求
根据附件中的代码， 补全 model 以及 调用 model 部分，参考论文1. 2. 

# 本次作业需要做 3个实验

 - 对100以内整数排序，待排序 序列长度为 5

 - 对100以内整数排序，待排序 序列长度为10

 - 对0~1之间的浮点数进行排序，序列长度为 5

# 本次报告包括

 - attention (可以参考 链接3仅供参考) （2分）

 - 介绍 pointer network ,以及 数字排序任务 （2分）

 - 实验 具体实现过程 （2分）

 - 实验的结果 分析 以及总结 （3分）

报告占 9分 ，代码 部分占 6 分。

# 参考

Pointer Networks

ORDER MATTERS: SEQUENCE TO SEQUENCE FOR SETS

https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html