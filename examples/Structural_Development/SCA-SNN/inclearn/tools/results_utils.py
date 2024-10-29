import glob
import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from . import utils


def get_template_results(cfg):
    return {"config": cfg, "results": []}


def save_results(results, label):
    del results["config"]["device"]

    folder_path = os.path.join("results", "{}_{}".format(utils.get_date(), label))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = "{}_{}_.json".format(utils.get_date(), results["config"]["seed"])
    with open(os.path.join(folder_path, file_path), "w+") as f:
        json.dump(results, f, indent=2)


def compute_avg_inc_acc(results):
    """Computes the average incremental accuracy as defined in iCaRL.

    The average incremental accuracies at task X are the average of accuracies
    at task 0, 1, ..., and X.

    :param accs: A list of dict for per-class accuracy at each step.
    :return: A float.
    """
    top1_tasks_accuracy = [r['top1']["total"] for r in results]
    top1acc = sum(top1_tasks_accuracy) / len(top1_tasks_accuracy)
    if "top5" in results[0].keys():
        top5_tasks_accuracy = [r['top5']["total"] for r in results]
        top5acc = sum(top5_tasks_accuracy) / len(top5_tasks_accuracy)
    else:
        top5acc = None
    return top1acc, top5acc