# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on one or more datasets."""






import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
from detectron.utils.io import cache_url
import detectron.utils.c2 as c2_utils

c2_utils.import_detectron_ops()
#c2_utils.import_custom_ops()


#utils.c2.import_custom_ops()
#utils.c2.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        '--gpu-id',
        dest='gpu_id',
        help='specific gpu to run on',
        default=0,
        type=int
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    cfg.TEST.WEIGHTS = cache_url(cfg.TEST.WEIGHTS, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
        logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
        time.sleep(10)

    run_inference(
        cfg.TEST.WEIGHTS,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True,
        gpu_id=args.gpu_id
    )
