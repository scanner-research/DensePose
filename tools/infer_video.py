# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import pickle
import numpy as np

from caffe2.python import workspace
import pycocotools.mask as mask_util

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils


c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--data-dir',
        dest='data_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--input-video',
        dest='input_video',
        help='input video file name',
        type=str
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def infer_one_frame(image, model, img_path, pose_path):
    with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                model, image, None, None
            )

    boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
            cls_boxes, cls_segms, cls_keyps)
    bodys = cls_bodys[1]

    valid_detection_inds = boxes[:, 4] > 0.9
    if boxes.shape[0] == 0 or np.count_nonzero(valid_detection_inds) < 2:
       return {}
    areas = mask_util.area(segms)
    top2_inds = np.argsort(areas * valid_detection_inds)[-2:]
    # decide foreground player    
    id1, id2 = top2_inds[0], top2_inds[1]
    top2_inds = [id2, id1] if boxes[id1][1] < boxes[id2][1] else [id1, id2]

    result = {'boxes': [boxes[i][:4].astype(int) for i in top2_inds], 'segms': [segms[i] for i in top2_inds], \
              'keyps': [keypoints[i].astype(int) for i in top2_inds], } #'bodys': [bodys[i] for i in top2_inds]}

    # crop foreground player
    body_box = boxes[top2_inds[0]][:4].astype(int) # x1, y1, x2, y2
    uv_patch = bodys[top2_inds[0]].transpose([1, 2, 0]) 
    uv_full = np.zeros(image.shape)
    uv_full[body_box[1] : body_box[1] + uv_patch.shape[0], body_box[0] : body_box[0] + uv_patch.shape[1], :] = uv_patch
    uv_full[:, :, 1:3] = 255. * uv_full[:, :, 1:3]
    uv_full[uv_full > 255] = 255.
    uv_full = uv_full.astype(np.uint8)

    cx, cy = (body_box[0] + body_box[2]) // 2, (body_box[1] + body_box[3]) // 2
    crop_width = 640 #image.shape[0] // 2
    crop_box = [cx - crop_width // 2, min(body_box[3] + 30, image.shape[0]) - crop_width]
    crop_box += [crop_box[0] + crop_width, crop_box[1] + crop_width]

    # body center box
    # crop_box = [cx - W // 2, cy - W // 2, W, W]
    if crop_box[0] < 0 or crop_box[2] > image.shape[1] or crop_box[1] < 0 or crop_box[3] > image.shape[0]
        return result
    
    # visualize crop
    # cv2.rectangle(image, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (255,255,255), 5)
    # cv2.rectangle(uv_full, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (255,255,255), 5)
    # cv2.imwrite(img_path, image)
    # cv2.imwrite(pose_path, uv_full)
    
    cv2.imwrite(img_path, image[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2], :])
    cv2.imwrite(pose_path, uv_full[crop_box[1] : crop_box[3], crop_box[0] : crop_box[2], :])
    result['crop_box'] = crop_box
    result['img_path'] = img_path   
    result['densepose_path'] = pose_path   
    return result


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()


    # load match scene
    match_scene_dict = pickle.load(open('{}/pkl/match_scene_intervals_dict.pkl'.format(args.data_dir), 'rb'))
    HW_foreground = set()
    JZ_foreground = set()
    for match in match_scene_dict['HW_foreground']:
        for fid in range(match[1], match[2]):
            HW_foreground.add(fid)
    for match in match_scene_dict['JZ_foreground']:
        for fid in range(match[1], match[2]):
            JZ_foreground.add(fid)

    # detect frame
    cap = cv2.VideoCapture('{}/videos/{}'.format(args.data_dir, args.input_video))
    result_list = []
    fid = 0
    ret = True

    # create dir
    for player in ['HW', 'JZ']:
        dir = '{data_dir}/image/img_{player}'.format(data_dir=args.data_dir, player=player)
        if not os.path.exists(dir):
            os.makedirs(dir)
        dir = '{data_dir}/image/densepose_{player}'.format(data_dir=args.data_dir, player=player)
        if not os.path.exists(dir):
            os.makedirs(dir)

    vid = 65
    while ret:
        if fid % 1000 == 0:
            print("Inferring frame %d" % fid)
        ret, image = cap.read()
        if fid in HW_foreground or fid in JZ_foreground:
            player = 'HW' if fid in HW_foreground else 'JZ'
            img_path = '{data_dir}/image/img_{player}/img_{vid}_{fid}_{player}.jpg'.format(data_dir=args.data_dir, player=player, vid=vid, fid=fid)
            pose_path = '{data_dir}/image/densepose_{player}/densepose_{vid}_{fid}_{player}.jpg'.format(data_dir=args.data_dir, player=player, vid=vid, fid=fid)

            result_one_image = infer_one_frame(image, model, img_path, pose_path)
            result_one_image['foreground'] = player
            result_one_image['fid'] = fid
            result_list.append(result_one_image)
        fid += 1
        # if len(result_list) > 100:
        #     break

    pickle.dump(result_list, open('{}/pkl/result.pkl'.format(args.data_dir), 'wb'), protocol=2)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
