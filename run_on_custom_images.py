import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

import numpy as np
from mmdet.core import get_classes
import pycocotools.mask as maskUtils

import json
import os
import glob
import time

import argparse
import magic

def gen_fnames(vid_frames):
    img_fnames = list()
    for n in range(len(vid_frames)):
        frame_fname = list('000000')
        frame_fname[-len(str(n))::] = str(n)
        frame_fname = str().join(frame_fname)
        img_fnames.append('{}.jpg'.format(frame_fname))
    return img_fnames

def get_class_bboxes(input_path, model, cfg, dataset='coco', class_int=0, score_thr=0.78, show_result=False):
    '''
    :param input_path:
    :param model:
    :param cfg:
    :param dataset:
    :param class_int:
    :param score_thr:
    :param show_result:
    :return:
    '''
    vid_frames = list()

    if os.path.isdir(input_path):
        img_fnames = glob.glob('{}/*.jpg'.format(input_path))
        img_size = mmcv.imread(img_fnames[0]).shape[:2]
        detections = inference_detector(model, img_fnames, cfg)
    elif os.path.isfile(input_path):
        mime = magic.from_file(input_path, mime=True)
        if 'image' in mime:
            img_fnames = [input_path]
            img_size = mmcv.imread(img_fnames[0]).shape[:2]
            detections = [inference_detector(model, input_path, cfg)]
        elif 'video' in mime:
            # Read video (iterator)
            video_it = mmcv.VideoReader(input_path)
            vid_frames = list(video_it)

            # auto-generate virtual file names to keep coherence
            img_fnames = gen_fnames(vid_frames)

            # img_fnames = ['{}.jpg'.format(n*step + 1) for n in range(len(np_images_vid))]

            img_size = video_it.resolution
            detections = inference_detector(model, vid_frames, cfg)
        else:
            raise Exception('Input file type not supported.')
    else:
        raise Exception('Input path does not exist.')


    class_names = get_classes(dataset)

    result_dict = dict()
    result_dict['image_size'] = img_size
    result_dict['results'] = dict()

    for idx, det in enumerate(list(detections)):
        if isinstance(det, tuple):
            bbox_result, segm_result = det
        else:
            bbox_result, segm_result = det, None

        bboxes = np.vstack(bbox_result)

        if bbox_result is not None:
            # segms = mmcv.concat_list(segm_result)

            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]

            labels = np.concatenate(labels)

            filter_thr = np.where(bboxes[:, -1] > score_thr)[0]
            filter_class = np.where(labels == class_int)[0]
            filter_idxs = np.intersect1d(filter_thr, filter_class)

            bboxes = bboxes[filter_idxs]
            labels = labels[filter_idxs]

            data = list()
            for bbox, label in zip(bboxes, labels):
                left_top = [int(bbox[0]), int(bbox[1])]
                right_bottom = [int(bbox[2]), int(bbox[3])]
                label_name = class_names[label] if class_names is not None else 'cls {}'.format(label)
                data.append({'label': label_name, 'bbox': {'lt': left_top, 'rb': right_bottom}})

            result_dict['results'][os.path.basename(img_fnames[idx])] = data.copy()
            data.clear()

            ## Debug
            if show_result:
                img = vid_frames[idx] if vid_frames else mmcv.imread(img_fnames[idx])
                mmcv.imshow_det_bboxes(
                    img.copy(),
                    bboxes,
                    labels,
                    class_names=class_names,
                    score_thr=score_thr,
                    show=show_result)
            ##

    with open('{}_detection_bboxes.json'.format(time.strftime("%Y%m%d%H%M%S")), 'w') as out_file:
        json.dump(result_dict, out_file)

    # print(json.dumps(out))  # debug
    return result_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", help="Path to directory containing images OR path to a single image",
                        required=True)
    parser.add_argument("-c", "--cls", default=0, help="Coco class index (int)")
    parser.add_argument("-sc", "--score", default=0.90, help="Score threshold (0-1)")
    parser.add_argument("-s", "--show", help="Show result", action='store_true')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile('configs/dcn/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x.py')
    cfg.model.pretrained = None

    # construct the model and load checkpoint
    model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
    _ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/dcn/cascade_mask_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-09d8a443.pth')

    out = get_class_bboxes(args.path, model, cfg, dataset='coco', class_int=args.cls, score_thr=args.score,
    show_result=args.show)


if __name__ == "__main__":
    main()
