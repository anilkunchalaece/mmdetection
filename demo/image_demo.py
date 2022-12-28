# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import os
from argparse import ArgumentParser
import mmcv

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input_dir', help="Input dir with images", default=None)
    parser.add_argument('--output_dir', help="output dir where to store images")
    parser.add_argument('--background_type', help="type of background, if this is dynamic, generate masks for all images else generate for first 100")
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--config', help='Config file',
                        default="configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py")
    parser.add_argument('--checkpoint', help='Checkpoint file',
                        default="checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth")
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    
    if args.input_dir == None: # if no directory mentioned
        # test a single image
        result = inference_detector(model, args.img)
    else :
        imgList = [os.path.join(args.input_dir,x) for x in sorted(os.listdir(args.input_dir),key= lambda x: int(x.split(".")[0]))]
        # Need all the masks, background is not really static
        #  if args.background_type == "static" :
        #     imgList = imgList[:160] # take first hundred
        
        os.makedirs(os.path.join(args.output_dir,'masks'),exist_ok=True)

        for idx, img in enumerate(mmcv.track_iter_progress(imgList)) :
            outFileName = os.path.join(args.output_dir,'masks',os.path.basename(imgList[idx]))
            result = inference_detector(model,img) #run the detector
            # show the results
            show_result_pyplot(
                model,
                img,
                result,
                palette=args.palette,
                score_thr=args.score_thr,
                out_file=outFileName)


async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
