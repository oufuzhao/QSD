import logging

import torch
import torch.nn as nn
from ignite.engine import Engine
import argparse
import os
import sys
from config.defaults import _C as cfg
from modeling import build_model
from torch.backends import cudnn
from data import make_data_loader

from utils.reid_metric import r1_mAP_mINP, r1_mAP_mINP_reranking

def create_supervised_evaluator(model, metrics, device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to evaluate
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, imgpath = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)[0]
            return feat, pids, camids, imgpath

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def do_test(cfg, model, data_loader, num_query):
    model.eval()
    device = cfg.MODEL.DEVICE
    if cfg.TEST.RE_RANKING == 'off':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    elif cfg.TEST.RE_RANKING == 'on':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP_mINP': r1_mAP_mINP_reranking(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device)
    else:
        print("Unsupported re_ranking config. Only support for on or off, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(data_loader['eval'])
    cmc, mAP, mINP = evaluator.state.metrics['r1_mAP_mINP']
    model.train()
    return cmc[0], mAP, mINP
