# encoding: utf-8

import collections
from loss.QSD import QSD_loss
from train import do_train
from test import do_test
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


def main(config_file):
    parser = argparse.ArgumentParser(description="Re-ID")
    parser.add_argument(
        "--config_file", default=config_file, help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    if args.config_file != "": cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("Re-ID", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda": os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    data_loader, num_query, num_classes = make_data_loader(cfg)

    print('loading the baseline model')
    model = build_model(cfg, num_classes)
    try: model_dict = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cuda').state_dict()
    except: model_dict = torch.load(cfg.MODEL.PRETRAIN_PATH, map_location='cuda')
    
    model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
    net_dict = model.state_dict()
    same_dict = {k: v for k, v in model_dict.items() if k in net_dict}
    diff_dict = {k: v for k, v in net_dict.items() if k not in model_dict}
    print(f"LOADING DONE LAYERS: {len(same_dict)}/{len(model_dict)}")
    ignore_dictName = list(diff_dict.keys())
    print('INGNORING LAYERS:')
    print(ignore_dictName)
    net_dict.update(same_dict)
    model.load_state_dict(net_dict)

    criterion = model.get_creterion(cfg, num_classes)
    loss_qsd = QSD_loss(distance='L2').cuda()
    optimizer = model.get_optimizer(cfg, criterion)

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
        start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')
                           [-1].split('.')[0].split('_')[-1])
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
        print('optimizer', optimizer)
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
        scheduler = WarmupMultiStepLR(optimizer['model'], cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD, start_epoch)
    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    if 'cpu' not in cfg.MODEL.DEVICE:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device=cfg.MODEL.DEVICE)

    if cfg.TEST.EVALUATE_ONLY == 'on':
        print("Evaluate Only")
        model.load_param(cfg.TEST.WEIGHT)
        r1, mAP, mINP = do_test(cfg, model, data_loader, num_query)
        print("Rank-1: {:.1%}".format(r1))
        print("mAP: {:.1%}".format(mAP))
        print("mINP: {:.1%}".format(mINP))
        print("{:.1%}\t{:.1%}\t{:.1%}".format(r1,mAP,mINP))   
        return

    logger.info("===> Baseline Testing")
    r1, mAP, mINP = do_test(cfg, model, data_loader, num_query)
    logger.info("Rank-1: {:.1%}".format(r1))
    logger.info("mAP: {:.1%}".format(mAP))
    logger.info("mINP: {:.1%}".format(mINP))
    logger.info("{:.1%}\t{:.1%}\t{:.1%}".format(r1,mAP,mINP))    

    do_train(cfg,
             model,
             data_loader,
             optimizer,
             scheduler,
             criterion,
             loss_qsd,
             num_query,
             start_epoch
             )

if __name__ == '__main__':
    config_file = "/root/autodl-nas/Re-ID/release_AGW/configs/AGW_QSD.yml"
    main(config_file)
