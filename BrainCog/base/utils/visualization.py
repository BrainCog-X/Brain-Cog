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

import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import seaborn as sns


# Random state.
RS = 20150101


def plot_tsne(x, colors, output_dir):
    """
    绘制t-SNE聚类图, 直接将图片保存到输出路径
    :param x: 输入的feature map / spike
    :param colors: predicted labels 作为不同类别的颜色
    :param output_dir: 图片输出的路径(包括图片名及后缀)
    :return: None
    """
    if isinstance(x, torch.Tensor):
        x = x.to('cpu').numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.to('cpu').numpy()

    x = TSNE(random_state=RS, n_components=2).fit_transform(x)
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    palette = np.array(sns.color_palette("hls", 10))
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
    plt.show()


def plot_tsne_3d(x, colors, output_dir):
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

    x = TSNE(random_state=RS, n_components=3, perplexity=30).fit_transform(x)
    # sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})
    fig = plt.figure(figsize=(8, 8))
    palette = np.array(sns.color_palette("hls", 10))
    ax = Axes3D(fig)
    sc = ax.scatter(x[:, 0], x[:, 1], x[:, 2], lw=0, s=20, alpha=0.8,
                    c=palette[colors.astype(np.int)])
    ax.view_init(elev=15, azim=30)
    # plt.xlim(-25, 25)
    # plt.ylim(-25, 25)
    # ax.axis('off')
    # ax.axis('tight')
    # plt.grid('off')

    plt.savefig(output_dir, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.show()


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
    plt.show()


if __name__ == '__main__':
    # Test for T-SNE
    # x = torch.randn((100, 100))
    # y = torch.randint(low=0, high=10, size=[100])
    # plot_tsne_3d(x, y, output_dir='./t-sne.eps')

    # Test for confusion matrix
    x = torch.rand(5012, 100)
    y = torch.randint(0, 100, (5012, ))
    plot_confusion_matrix(x, y, '')
