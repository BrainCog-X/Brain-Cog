# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2022/7/1 11:10
# User      : Floyed
# Product   : PyCharm
# Project   : braincog
# File      : visualization.py
# explain   : add t-SNE

import os
import numpy as np
import sklearn
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from einops import rearrange
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import seaborn as sns

# Random state.
RS = 20150101



def spike_rate_vis_1d(data, output_dir=''):
    assert len(data.shape) == 2, 'Shape should be (t, c).'

    data = rearrange(data, 'i j -> j i')
    if isinstance(data, torch.Tensor):
        data = data.to('cpu').numpy()

    plt.figure(figsize=(8, 8))
    sns.heatmap(data, annot=None, cmap='YlGnBu')
    # plt.ylim(0, _max + 1)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def spike_rate_vis(data, output_dir=''):
    assert len(data.shape) == 3, 'Shape should be (t, r, c).'
    data = data.mean(axis=0)

    if isinstance(data, torch.Tensor):
        data = data.to('cpu').numpy()

    plt.figure(figsize=(8, 8))
    sns.heatmap(data, annot=None, cmap='YlGnBu')
    # plt.ylim(0, _max + 1)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def plot_mem_distribution(data,
                          output_dir='',
                          legend='',
                          xlabel='Membrane Potential',
                          ylabel='Density',
                          **kwargs):
    # print(type(data), len(data))
    if isinstance(data, torch.Tensor):
        data = data.reshape(-1).to('cpu').numpy()

    mean = data.mean()
    std = data.std()
    idx = np.argwhere(data < mean - 3 * std)
    data = np.delete(data, idx)
    idx = np.argwhere(data > mean + 3 * std)
    data = np.delete(data, idx)
    
    sns.set_style('darkgrid')
    # sns.set_palette('deep', desat=.6)
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
 
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, aspect='equal')
         
    # sns.distplot(data, bins=int(np.sqrt(data.shape[0])),
    #              hist=True, kde=False, hist_kws={'histtype': 'stepfilled'}, **kwargs)

    # print('hist begin')
    print(len(data))
    n, bins, patches = plt.hist(data,
                                density=True,
                                histtype='stepfilled',
                                alpha=0.618,
                                bins=int(np.sqrt(data.shape[0])),
                                **kwargs)
    # print('hist finished')
    # sns.kdeplot(data, color='#5294c3')
    # print('kde finished')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # if legend != '':
    #     plt.legend(legend)
    # ax.axis('tight')

    if output_dir != '':
        plt.savefig(output_dir, bbox_inches='tight')
        print('{} saved'.format(output_dir))
    # plt.show()


def plot_tsne(x, colors,output_dir="", num_classes=None):
    if isinstance(x, torch.Tensor):
        x = x.to('cpu').numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.to('cpu').numpy()

    if num_classes is None:
        num_classes=colors.max()+1
    x = TSNE(random_state=RS, n_components=2).fit_transform(x)
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    palette = np.array(sns.color_palette("hls", num_classes))
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=25,
                    c=palette[colors.astype(np.int)])
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    ax.axis('tight')
    # plt.grid('off')

    plt.savefig(output_dir, facecolor=fig.get_facecolor(), bbox_inches='tight')
    #plt.show()


def plot_tsne_3d(x, colors,output_dir="", num_classes=None):
    """
    绘制3D t-SNE聚类图, 直接将图片保存到输出路径
    :param x: 输入的feature map / spike
    :param colors: predicted labels 作为不同类别的颜色
    :param output_dir: 图片输出的路径(包括图片名及后缀)
    :return: None
    """
    if isinstance(x, torch.Tensor):
        x = x.to('cpu').numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.to('cpu').numpy()

    if num_classes is None:
        num_classes=colors.max()+1
    x = TSNE(random_state=RS, n_components=3, perplexity=30).fit_transform(x)
    # sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    fig = plt.figure(figsize=(8, 8))

    palette = np.array(sns.color_palette("hls", num_classes))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], lw=0, s=20, alpha=0.8,
                    c=palette[colors.astype(np.int)])

    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.view_init(20, -120)
    ax.axis('tight')
    plt.savefig(output_dir, facecolor=fig.get_facecolor(), bbox_inches='tight')
    #plt.show()


def plot_confusion_matrix(logits, labels, output_dir):
    """
    绘制混淆矩阵图
    :param logits: predicted labels
    :param labels: true labels
    :param output_dir: 输出路径, 需要包括文件名以及后缀
    :return: None
    """
    sns.set_style('darkgrid')
    sns.set_palette('Blues_r')
    sns.set_context("notebook", font_scale=1.,
                    rc={"lines.linewidth": 2.})

    logits = logits.argmax(dim=1).cpu()
    labels = labels.cpu()
    _max = labels.max()
    if _max > 10:
        annot = False
    else:
        annot = True
    # print(labels.shape, logits.shape)
    conf_matrix = confusion_matrix(labels, logits)
    con_mat_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)
    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm, annot=annot, cmap='Blues')
    plt.ylim(0, _max + 1)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')

    plt.savefig(output_dir, bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':
    # Test for T-SNE
    # x = torch.randn((100, 100))
    # y = torch.randint(low=0, high=10, size=[100])
    # plot_tsne_3d(x, y, output_dir='./t-sne.eps')

    # Test for confusion matrix
    # x = torch.rand(5012, 100)
    # y = torch.randint(0, 100, (5012,))
    # plot_confusion_matrix(x, y, '')

    # Test for Mem Distribution
    x = torch.randn(100000)
    plot_mem_distribution(x, legend=['test'])

