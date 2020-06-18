import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import config

def pl():
    plt.ion()
    plt.plot([1,2,3,3],[2,3,4,3])
    plt.scatter([322,342,34],[34,4,3] ,color=config.color[7])

    plt.scatter([32,34,34],[34,4,3], color=config.color[4])
    plt.scatter([12,30,24],[44,24,3] ,color=config.color[2])
    plt.ioff()
    plt.show()

def bar_figure():
    x = [1,2,3,4,5]
    y = [76,36,97,88,67]
    plt.bar(x,y)
    plt.show()


if __name__ == '__main__':
    
    bar_figure()
  
#我是笔记本啊
#2070
