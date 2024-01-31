# encoding: utf-8

import collections
from modeling_q import build_model_q
from config_q import cfg_q
from loss.quality_KD import KD_loss
from tools.test import do_test
from tools.train import do_train
from utils.logger import setup_logger
from utils.lr_scheduler import WarmupMultiStepLR
from modeling import build_model
from data import make_data_loader
from config.defaults import _C as cfg
import argparse
import os
import sys
import torch

from torch.backends import cudnn

sys.path.append('.')


torch.nn.Module.dump_patches = True


def main():
    parser = argparse.ArgumentParser(description="AGW Re-ID Baseline")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]
                   ) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        # new add by gu
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    data_loader, num_query, num_classes = make_data_loader(cfg)

#    for data_item in data_loader['train']:
#        print('len(data_item)',len(data_item))
#        print('data_item',data_item)
#        sys.exit()

    print('loading model_q')
    model = build_model(cfg, num_classes)
    model_q = build_model_q(cfg_q)
    model_q_dict = torch.load('/root/autodl-nas/Re-ID/Re-ID-AGW/checkloints/dekeMTMC/dukeMTMC_QA_model.pth')
    model_q_dict = {k.replace('module.', ''): v for k, v in model_q_dict.items()}
    net_dict = model_q.state_dict()
    same_dict = {k: v for k, v in model_q_dict.items() if k in net_dict}
    diff_dict = {k: v for k, v in net_dict.items() if k not in model_q_dict}
    print(f"LOADING DONE LAYERS: {len(same_dict)}/{len(model_q_dict)}")
    ignore_dictName = list(diff_dict.keys())
    # LOG.record('='*20 + 'INGNORING LAYERS' + '='*20)
    print('INGNORING LAYERS:')
    print(ignore_dictName)
    model_q.eval()

    criterion = model.get_creterion(cfg, num_classes)
    loss_kd = KD_loss(distance='L2').cuda()
    optimizer = model.get_optimizer(cfg, criterion)

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')
                           [-1].split('.')[0].split('_')[-1])
#        start_epoch = 1
        print('Start epoch:', start_epoch)
        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace(
            'model', 'optimizer')
        print('Path to the checkpoint of optimizer:', path_to_optimizer)
        path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace(
            'model', 'center_param')
        print('Path to the checkpoint of center_param:', path_to_center_param)
        path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace(
            'model', 'optimizer_center')
        print('Path to the checkpoint of optimizer_center:',
              path_to_optimizer_center)
#        model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
#        checkpoint=torch.load(path_to_optimizer)
#        print(checkpoint)
#        for key in checkpoint.keys():
#            print(key)
        print('optimizer', optimizer)
#        print('optimizer[model]',optimizer['model'])
#        optimizer_checkpoint=torch.load(path_to_optimizer)
#        optimizer['model'].load_state_dict(optimizer_checkpoint)
        criterion['center'].load_state_dict(torch.load(path_to_center_param))
        optimizer['center'].load_state_dict(
            torch.load(path_to_optimizer_center))
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    elif cfg.MODEL.PRETRAIN_CHOICE == 'latest':
        start_epoch = 110
#        start_epoch = 1
#        print('Start epoch:', start_epoch)
#        path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
#        print('Path to the checkpoint of optimizer:', path_to_optimizer)
#        path_to_center_param = cfg.MODEL.PRETRAIN_PATH.replace('model', 'center_param')
#        print('Path to the checkpoint of center_param:', path_to_center_param)
#        path_to_optimizer_center = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer_center')
#        print('Path to the checkpoint of optimizer_center:', path_to_optimizer_center)
        print('cfg.MODEL.PRETRAIN_PATH', cfg.MODEL.PRETRAIN_PATH)
        param_dict = torch.load(cfg.MODEL.PRETRAIN_PATH).module
#        print(param_dict)
        if not isinstance(param_dict, collections.OrderedDict):
            param_dict = param_dict.state_dict()
#        for key in param_dict.keys():
#            print(key)
        model.load_state_dict(param_dict)
#        model.load_param(cfg.MODEL.PRETRAIN_PATH)
#        checkpoint=torch.load(path_to_optimizer)
#        print(checkpoint)
#        for key in checkpoint.keys():
#            print(key)
#        print('optimizer',optimizer)
# print('optimizer[model]',optimizer['model'])
# optimizer_checkpoint=torch.load(path_to_optimizer)
# optimizer['model'].load_state_dict(optimizer_checkpoint)
#        criterion['center'].load_state_dict(torch.load(path_to_center_param))
#        optimizer['center'].load_state_dict(torch.load(path_to_optimizer_center))
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)

    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(
            cfg.MODEL.PRETRAIN_CHOICE))

    if 'cpu' not in cfg.MODEL.DEVICE:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            model_q = torch.nn.DataParallel(model_q)
        model.to(device=cfg.MODEL.DEVICE)
        model_q.to(device=cfg.MODEL.DEVICE)

    if cfg.TEST.EVALUATE_ONLY == 'on':
        logger.info("Evaluate Only")
        model.load_param(cfg.TEST.WEIGHT)
        do_test(cfg, model, data_loader, num_query)
        return

    do_train(cfg,
             model,
             model_q,
             data_loader,
             optimizer,
             scheduler,
             criterion,
             loss_kd,
             num_query,
             start_epoch
             )


if __name__ == '__main__':
    main()
