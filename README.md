# weights_autosave
Weights_Autosave
    
    To run:
    usage: weights_autosave.py [-h] [--weight_dir WEIGHT_DIR]
                               [--weight_prefix WEIGHT_PREFIX]
                               [--weight WEIGHT_FIXED] [--save_dir SAVE_DIR]
                               [--detector_cmd DETECTOR_CMD]
                               [--iou_ratio IOU_RATIO] [--period PERIOD]
                               [--nbest NBEST] [--log LOG] [--verbose VERBOSE]
                               
    Auto-save the weight file when training yolo
    
    optional arguments:
      -h, --help            show this help message and exit
      --weight_dir WEIGHT_DIR, -wdir WEIGHT_DIR, --wdir WEIGHT_DIR
                            weight file directory when training
      --weight_prefix WEIGHT_PREFIX, -weight_prefix WEIGHT_PREFIX, --pre WEIGHT_PREFIX, -pre WEIGHT_PREFIX
                            prefix of weight files,the naming convention of the
                            weight files must be ${weight_prefix}_d+.weights for different weights
      --weight WEIGHT_FIXED, -weight WEIGHT_FIXED, --w WEIGHT_FIXED, -w WEIGHT_FIXED
                            single weight file name which will be keep
                            updated,auto-saving will keep checking on this weight file.
      --save_dir SAVE_DIR, -save_dir SAVE_DIR, --sdir SAVE_DIR, -sdir SAVE_DIR
                            save the best weight file to this directory
      --detector_cmd DETECTOR_CMD, -detector_cmd DETECTOR_CMD, --detector DETECTOR_CMD, -detector DETECTOR_CMD
                            The darknet detector map cmd to check the IoU and mAP
                            of weight in training.
      --iou_ratio IOU_RATIO, -iou_ratio IOU_RATIO, --iou IOU_RATIO, -iou IOU_RATIO
                            IoU ratio for best weight scoring,the map_ratio =
                            (1.0-iou_ratio),the score is (iou_ratio*IoU + map_ratio*mAP)
      --period PERIOD, -period PERIOD, --p PERIOD, -p PERIOD
                            check the weight status in every <period> minutes
      --nbest NBEST, -nbest NBEST, --nb NBEST, -nb NBEST
                            save the best <nbest> weight files,default 5.
      --log LOG, -log LOG   log file to save the logs
      --verbose VERBOSE, -verbose VERBOSE, --v VERBOSE, -v VERBOSE
                            show debug log
