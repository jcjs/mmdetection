import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

import numpy as np
from mmdet.core import get_classes
import pycocotools.mask as maskUtils

# person class idx: 0
def get_class_boxes(img, result, dataset='coco', class_int=0, score_thr=0.7, show_result=False):
    img = mmcv.imread(img)
    class_names = get_classes(dataset)

    if isinstance(result, tuple):
        bbox_result, segm_result = result
    else:
        bbox_result, segm_result = result, None

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

        res = list()
        for bbox, label in zip(bboxes, labels):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            label_name = class_names[label] if class_names is not None else 'cls {}'.format(label)
            res.append({'label': label_name, 'bbox': {'lt': left_top, 'rb': right_bottom}})

        if show_result:
            out_file = None
            mmcv.imshow_det_bboxes(
                img.copy(),
                bboxes,
                labels,
                class_names=class_names,
                score_thr=score_thr,
                show=out_file is None)
        print(res)
        return res


cfg = mmcv.Config.fromfile('configs/faster_rcnn_r50_fpn_1x.py')
# cfg = mmcv.Config.fromfile('configs/faster_rcnn_x101_64x4d_fpn_1x.py')

cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
# _ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pth')


# test a single image
img = mmcv.imread('test_images/00000001.jpg')
result = inference_detector(model, img, cfg)

res = get_class_boxes(img, result, dataset='coco', class_int=0, score_thr=0.8, show_result=True)


# test a list of images
#imgs = ['test1.jpg', 'test2.jpg']
#for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#        print(i, imgs[i])
#            show_result(imgs[i], result)
