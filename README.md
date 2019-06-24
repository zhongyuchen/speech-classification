# speech-classification

思路：输入语音信号，获取特征，特征进入分类器分类，输出结果

语音信号要做端点检测，找出开始、结束点，裁剪好，再获取特征
说的快慢不一，用固定的窗取出的特征不一样，怎么处理？
MFCC：提取特征feature

分帧就是把连续的若干个点（256/512？）设为一帧，帧长一般取为 20 ~ 50 毫秒，20、25、30、40、50 都是比较常用的数值，甚至还有人用 32
相邻两帧的起始位置的时间差叫做帧移，常见的取法是取为帧长的一半

核心代码自己写，不要全都调库->自己写的函数提出的特征，和用库提出的特征有多少区别？自己的和库的代码有多少区别？

1. 用包提取特征，训一个cnn网络（pre-train），看下acc，确认这个网络是可用的
2. 自己提取特征，重新训cnn网络（pre-train），看最终acc

sh 脚本文件+argparse/tf的flags 
或者 
configparser(path management included)+ini file

create model?
pytorch/tensorflow template?


## 数据的问题

1.
多余文件
16307130173.textClipping
命名错误
16307130302-15-19 (1).wav
16307130343-15-18 (1).wav
缺少文件
16307130343-05-18.wav

总共有13599份文件
2. 
_wave_params(nchannels=1, sampwidth=2, framerate=16000, nframes=29721, comptype='NONE', compname='not compressed')
只有nframes不同，15种


1.
8k sampling rate -> 256 帧each window
16k -> 512 帧each window

2.
切窗口，把信号两端的噪声切掉，越精准越好：短时平均能量（适合噪声小）、短时平均过零率（适合噪声大）
可以两个量一起来判断：（一些参数要自己调整）
    先做能量，再做过零率（认为能量和过零率是独立的）
    一起做（先求出能量和过零率的二维联合分布，再判断<-概率密度函数的参数估计，用多个样本的真实起始点来估计概率密度函数的参数，会很累）
    
## Start

13599->11599+2000

python run_cnn.py --model vgg --data mfcc
