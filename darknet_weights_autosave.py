#!/home/b/.conda/envs/pytorch/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 09 20:34:37  

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

@author: Administrator
"""

#######################################
# import modules
import os
import sys
import logging
import time
from logging import debug,info,warning,error
from datetime import datetime
from collections import defaultdict
import argparse
import re
import json
import pandas as pd
import copy
import numpy as np

#######################################
# global variables
# run cmd of detector
#DETECTOR_CMD = 'darknet detector map ../../darknet.q/cfg/voc-tiny.data ../../darknet.q/cfg/yolov3-tiny-voc.cfg'
#DETECTOR_CMD = './darknet detector map ./cfg/voc-tiny.data ./cfg/yolov3-tiny-voc.cfg '
DETECTOR_CMD = '/home/aiden00/darknet/darknet detector map /home/aiden00/darknetAnlogic/cfg/voc-tiny/voc-tiny.data /home/aiden00/darknetAnlogic/cfg/voc-tiny/yolov3-tiny-voc_training.cfg.prune.test '
BEST_WEIGHT_PREFIX  = 'yolov3-tiny-voc.weight' # best weight name

#######################################
# body
DATELOGO = datetime.now().strftime('%Y%m%d_%H%M%S')

def dumpjson(data,jsfile,mode='w+'):
    '''
    dump json
        jsfile: save to jsfile, if None , print it
    '''
    if not jsfile:
        try:
            js = json.dumps(data,sort_keys=True,indent=4,separators=(',',': '))
            print(js)
        except:
            error('[FAIL]: Dump json failed.')
    else:
        try:
            with open(jsfile,mode) as fout:
                json.dump(data,fout,sort_keys=True,indent=4,separators=(',',': '))
        except:
            error('[FAIL]: Dump json file %s failed.' % jsfile)



class c_SaveWeights(object):
    '''
    Check the weight file status in every <period> minutes,
    Using detector map to find the IoU and mAP, calculate the score IoU * iou_ratio + mAP * map_ratio
    and saving the best score weight.
    '''
    def __init__(self,args):
        '''
        '''
        self.weight_dir         = args.weight_dir
        self.weight_prefix      = args.weight_prefix
        self.weight_fixed       = args.weight_fixed
        self.save_dir           = os.path.abspath(args.save_dir)
        self.detector_cmd       = args.detector_cmd
        self.iou_ratio          = args.iou_ratio
        self.period             = args.period
        self.nbest              = args.nbest
        #
        self.map_ratio          = 1.0 - args.iou_ratio
        self.start              = time.time() # in seconds
        self.period             = int(self.period * 60) # minutes to seconds
        self.scores             = defaultdict(dict) # score: {weight:{map:,iou:,score:,saved:}}
        self.wbest_prefix       = os.path.join(self.save_dir,BEST_WEIGHT_PREFIX)
        self.current_time       = '' #
        self.df_scores          = None
        self.saved_json         = os.path.join(self.save_dir,'weights.json')
        self.check_json         = True
        self.best_scores        = defaultdict(dict)
        self.saved_best_json    = os.path.join(self.save_dir,'weights.%sbest.json' % self.nbest)
        self.current_weight     = None
        self.cols               = ['score','IoU', 'mAP', 'iou_ratio', 'map_ratio','saved','weight','tag']
        self.it                 = 0
        self.best_saved         = [] # list of saved best weight files

        if len(self.weight_prefix):
            self.weight_fixed = "" # weight with prefix, not the fixed name

        info('# --------------------------------------------------------------')
        info('# Auto-saving weight files:')
        info('#   Run dir     :     %s' % os.getcwd())
        info('#   Period      :     %s sec' % self.period)
        info('#   Weight dir  :     %s' % self.weight_dir )
        info('#   Weight tag  :     %s' % self.weight_prefix) if self.weight_prefix else ""
        info('#   Weight tag  :     %s' % self.weight_fixed ) if self.weight_fixed  else ""
        info('#   Save to dir :     %s' % self.save_dir)
        info('#   Score       :     IoU * %.3f + mAP * %.3f ' % (self.iou_ratio,self.map_ratio))
        info('#   Detector    :     %s' % self.detector_cmd)
        info('# --------------------------------------------------------------')

        cmd = 'mkdir -p %s' % self.save_dir
        try:
            os.system(cmd)
        except:
            error("Fail to execute cmd %s" % cmd)

    def load_weight_hist(self):
        '''
        '''
        pass

    def find_weights(self):
        '''
        '''
        f_weight = [os.path.join(self.weight_dir,f) for f in os.listdir(self.weight_dir)]
        if len(self.weight_fixed):
            f_weight = [os.path.join(self.weight_dir,self.weight_fixed)]
        elif len(self.weight_prefix):
            f_weight = [f for f in f_weight if re.findall(self.weight_prefix + '\d+.weights$',f)] 
        else:
            f_weight = []
            error("Can't find any weight file with prefix:%s or fixed name:%s" % (self.weight_prefix,self.weight_fixed))

        if not len(f_weight):
            warning("Can't find weight file under %s" % self.weight_dir)
        return f_weight

    def detecting(self,f_weight):
        '''
        '''
        self.current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        debug('f_weight=%s' % f_weight)
        for weight in f_weight:
            basename = os.path.basename(weight)
            tag = '%s.%s' % (basename,self.current_time) if len(self.weight_fixed) else basename
            #tag = '%s.%s' % (basename,self.current_time)
            #if weight in self.scores:
            if tag in self.scores:
                # skip detecting the weight file which have already been checked
                continue
            cmd = '%s %s' % (self.detector_cmd,weight)
            info("Scoring the weight %s @ %s" % (weight,self.current_time))
            try:
                #soso#
                result  = '''
                class_id = 0, name = aeroplane, 	 ap = 26.75 % 
                class_id = 1, name = bicycle, 	 ap = 30.70 % 
                class_id = 2, name = bird, 	 ap = 17.93 % 
                class_id = 3, name = boat, 	 ap = 25.78 % 
                class_id = 4, name = bottle, 	 ap = 15.56 % 
                class_id = 5, name = bus, 	 ap = 21.80 % 
                class_id = 6, name = car, 	 ap = 34.18 % 
                class_id = 7, name = cat, 	 ap = 12.84 % 
                class_id = 8, name = chair, 	 ap = 22.67 % 
                class_id = 9, name = cow, 	 ap = 36.05 % 
                class_id = 10, name = diningtable, 	 ap = 18.49 % 
                class_id = 11, name = dog, 	 ap = 12.90 % 
                class_id = 12, name = horse, 	 ap = 23.18 % 
                class_id = 13, name = motorbike, 	 ap = 26.85 % 
                class_id = 14, name = person, 	 ap = 33.30 % 
                class_id = 15, name = pottedplant, 	 ap = 12.67 % 
                class_id = 16, name = sheep, 	 ap = 43.44 % 
                class_id = 17, name = sofa, 	 ap = 22.68 % 
                class_id = 18, name = train, 	 ap = 28.13 % 
                class_id = 19, name = tvmonitor, 	 ap = 39.65 % 
                 for thresh = 0.25, precision = 0.41, recall = 0.28, F1-score = 0.34 
                 for thresh = 0.25, TP = 3425, FP = 4879, FN = 8607, average IoU = 27.10 % 
                
                 mean average precision (mAP) = 0.252775, or 25.28 % 
                '''
                result = ''
                result = os.popen(cmd).read()
                #soso#
                info(result)
            except:
                error("Fail to execute cmd %s" % cmd)
                continue
            #check if core dump
            if re.findall('core dump',result):
                error("Core dump! Exit the auto-save @ %s!" % self.current_time)
                sys.exit()

            # scoring
            #  for thresh = 0.25, precision = 0.00, recall = 0.00, F1-score = 0.00 
            #  for thresh = 0.25, TP = 2, FP = 35181, FN = 12030, average IoU = 0.00 % 
            #  mean average precision (mAP) = 0.000039, or 0.00 % 
            # Total Detection Time: 61.000000 Seconds
            try:
                IoU = float(re.findall('average IoU = (\d+\.\d+)',result)[0]) * 0.01
            except:
                error("Can't find IoU...")
                IoU = -99999999
            try:
                mAP = float(re.findall('\(mAP\) = (\d+\.\d+)',result)[0])
            except:
                error("Can't find mAP...")
                mAP = -99999999
            ####soso#
            ##IoU = np.random.uniform()
            ##mAP = np.random.uniform()
            ####soso#
            score = self.iou_ratio * IoU + self.map_ratio * mAP
            info('@ %s IoU=%.4f mAP=%.4f score=%.4f weight:%s' % (self.current_time,IoU,mAP,score,weight))
            self.scores[tag]['IoU']         = IoU
            self.scores[tag]['mAP']         = mAP
            self.scores[tag]['iou_ratio']   = self.iou_ratio
            self.scores[tag]['map_ratio']   = self.map_ratio
            self.scores[tag]['saved']       = ''
            self.scores[tag]['score']       = score
            self.scores[tag]['weight']      = weight # weight file 
            self.scores[tag]['tag']         = tag

    def scores_to_pd(self):
        '''
        scores dict to pandas
        '''
        # sort by pandas
        self.df_scores = pd.DataFrame([],columns=self.cols)
        for w in self.scores:
            self.df_scores = self.df_scores.append(pd.DataFrame([self.scores[w]],columns=self.cols),ignore_index=True)
        # the best nbest
        self.df_scores = self.df_scores.sort_values(by=['score'],ascending=False)

    def update_best(self):
        '''
        mv the xxx.best.<i>.tmp to xx.best.<i>
        '''
        cmds = []
        for i in range(self.nbest):
            saved = '%s.best.%s' % (self.wbest_prefix,i)  # 0 is the best
            tmp   = saved + '.tmp'
            cmd   = 'mv %s %s' % (tmp,saved)
            if os.path.exists(tmp):
                try:
                    os.system(cmd)
                    debug(cmd)
                except:
                    error('Fail to execute cmd %s' % cmd)
                    return False
                tag2 = os.path.basename(saved)
                tag2_tmp = tag2 + '.tmp'
                try:
                    self.scores[tag2] = copy.deepcopy(self.scores[tag2_tmp])
                except:
                    error("Can't transfer score from %s to %s" % (tag2,tag2_tmp))
                    return False
                self.scores[tag2]['saved'] = ''
                self.scores[tag2]['weight'] = saved
                self.scores[tag2]['tag'] = tag2
                del self.scores[tag2_tmp]

    def saving(self):
        '''
        '''
        self.scores_to_pd()
        # the best
        nbest_scores = self.df_scores[self.df_scores['saved'] == ""][:self.nbest]
        if nbest_scores.shape[0] == 0:
            warning("Can't find the best weight.")
            return
        # nbest_scores index to 0,1,...nbest-1
        idx  = list(nbest_scores.index)
        idx_rename = {idx[i]:i for i in range(len(idx))}
        debug('rename pre:%s new index:%s ' % (idx,idx_rename))
        nbest_scores = nbest_scores.rename(index=idx_rename)
        # find nbest
        #not_saved = nbest_scores[nbest_scores['saved'] == ""]
        # new weight file not inside the nbest
        tags = list(nbest_scores['tag'])
        idx  = list(nbest_scores.index)
        debug('tags=%s idx=%s' % (tags,idx))
        debug('weight=%s' % list(nbest_scores['weight']))
        debug('saved=%s' % list(nbest_scores['saved']))
        debug('scores:%s' % json.dumps(self.scores,sort_keys=True,indent=4,separators=(',',': ')) )
        # debug('nbest_scores:%s' % json.dumps(nbest_scores.to_dict(),sort_keys=True,indent=4,separators=(',',': ')) )

        # weight -> saved(tag2)
        for i,tag in enumerate(tags):
            saved = '%s.best.%s' % (self.wbest_prefix,idx[i])  # 0 is the best
            tag2 = os.path.basename(saved)
            debug('check index:%s tag:%s tag2:%s' % (idx[i],tag,tag2))
            if tag2 not in self.best_saved:
                self.best_saved.append(tag2)
            saved_tmp = saved + '.tmp'
            if tag2 == tag:
                # still keep the same order
                debug('Skip %s since its still keep the same order.' % saved)
                continue
            else:
                # cp to tmp first
                if tag in self.best_saved:
                    cmd = 'mv  %s %s' %(self.scores[tag]['weight'],saved_tmp)
                else:
                    cmd = 'cp -rf %s %s' %(self.scores[tag]['weight'],saved_tmp)
                try:
                    os.system(cmd)
                    debug(cmd)
                except:
                    error('Fail to execute cmd %s' % cmd)
                    return False

                # update self.scores[tag2]
                # tag -> tag2_tmp
                tag2_tmp = tag2 + '.tmp'
                if tag not in self.best_saved:
                    self.scores[tag]['saved'] = saved
                self.scores[tag2_tmp] = copy.deepcopy(self.scores[tag])

        #best.<i>.tmp -> best.<i>
        #tag2_tmp     -> tag2
        self.update_best()
        self.save_json()

    def save_json(self):
        # dump score
        if self.check_json and os.path.exists(self.saved_json):
            js2 = re.sub('.json$','.'+self.current_time+'.json',self.saved_json)
            cmd = 'cp -rf %s %s' %(self.saved_json,js2)
            try:
                info("The weights json file %s already exist, backup to %s" % (self.saved_json,js2))
                os.system(cmd)
                debug(cmd)
            except:
                error('Fail to execute cmd %s' % cmd)
        dumpjson(self.scores,self.saved_json,mode='w+')
        self.best_scores = {k:self.scores[k] for k in self.scores if k in self.best_saved}
        dumpjson(self.best_scores,self.saved_best_json,mode='w+')
        self.check_json = False

    def loop(self):
        '''
        '''
        while True:
            info('Weight auto-saving iteration %s started...' % self.it)
            self.it += 1
            f_weight = self.find_weights()
            self.detecting(f_weight)
            self.saving()
            info('Sleeping %s seconds' % self.period)
            time.sleep(self.period)

    def run(self):
        '''
        run the auto-saving
        '''
        self.load_weight_hist()
        self.loop()

def PyLogging(args = '',debugmode=False):
    '''
    basic Logging
    '''
    global LOGGING_LEVEL
    loglevel = logging.INFO
    if args:
        #if 'args.log' in locals().keys() and args.log
        loglevel = logging.DEBUG if args.verbose else logging.INFO
        LOGGING_LEVEL = 'debug' if args.verbose else 'info'
        if not args.verbose:
            if not args.log:
                logging.basicConfig(format='[%(asctime)s]:[%(levelname)s]:%(message)s',level=loglevel)
            else:
                logging.basicConfig(format='[%(asctime)s]:[%(levelname)s]:%(message)s',level=loglevel,filename=args.log,filemode='w')
        else:
            if not args.log:
                logging.basicConfig(format='[%(asctime)s]:[%(levelname)s]:[%(module)s.%(funcName)s]:%(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p',level=loglevel)
            else:
                logging.basicConfig(filename=args.log,filemode='w+',format='[%(asctime)s]:[%(levelname)s]:[%(module)s.%(funcName)s]:%(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p',level=loglevel)
        #logging.basicConfig(format='[%(levelname)s]:%(message)s',level=logging.INFO)
    if debugmode:
        loglevel = logging.DEBUG
        LOGGING_LEVEL = 'debug'
        logging.basicConfig(format='[%(asctime)s]:[%(levelname)s]:[%(module)s.%(funcName)s]:%(message)s', \
            datefmt='%m/%d/%Y %I:%M:%S %p',level=loglevel)
    # print to console
    if args.log:
        console = logging.StreamHandler()
        console.setLevel(loglevel)
        formatter = logging.Formatter('[%(asctime)s]:[%(levelname)s]:%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    #

def arg_parse():
    """
    Parse arguements to the weights autosave
    
    """
    parser = argparse.ArgumentParser(description='Auto-save the weight file when training yolo')
   
    parser.add_argument("--weight_dir", "-wdir", "--wdir",dest = 'weight_dir', help = "weight file directory when training",
                        default = "/home/aiden00/darknetAnlogic/backup_tiny_voc", type = str)
    parser.add_argument("--weight_prefix", "-weight_prefix", "--pre", "-pre", dest = 'weight_prefix', 
                        help = "prefix of weight files,the naming convention of the weight files must be ${weight_prefix}_d+.weights for different weights",
                        default = "", type = str)
    parser.add_argument("--weight", "-weight", "--w", "-w", dest = 'weight_fixed', 
                        help = "single weight file name which will be keep updated,auto-saving will keep checking on this weight file.",
                        default = "yolov3-tiny-voc_training.backup", type = str) 
    parser.add_argument("--save_dir", "-save_dir", "--sdir", "-sdir", dest = 'save_dir', help = "save the best weight file to this directory",
                        default = "./best_weights", type = str )
    parser.add_argument("--detector_cmd", "-detector_cmd", "--detector", "-detector", dest = 'detector_cmd', 
                        help = "The darknet detector map cmd to check the IoU and mAP of weight in training.",
                        default = DETECTOR_CMD, type = str )
    parser.add_argument("--iou_ratio", "-iou_ratio", "--iou", "-iou", dest = 'iou_ratio', 
                        help = "IoU ratio for best weight scoring,the map_ratio = (1.0-iou_ratio),the score is (iou_ratio*IoU + map_ratio*mAP) ",
                        default = "0.3", type = float )
    parser.add_argument("--period", "-period","--p","-p", dest = 'period', 
                        help = "check the weight status in every <period> minutes",
                        default = 5,type = float)
    parser.add_argument("--nbest", "-nbest","--nb","-nb", dest = 'nbest', 
                        help = "save the best <nbest> weight files,default 5.",
                        default = 5,type = int)
    parser.add_argument("--log", "-log", dest = 'log', 
                        help = "log file to save the logs",
                        default = "log.weights_autosave.%s.log" % DATELOGO)
    parser.add_argument("--verbose", "-verbose","--v","-v", dest = 'verbose', 
                        help = "show debug log",action="store_true",
                        default = False)
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = arg_parse()
    PyLogging(args)

    saver = c_SaveWeights(args)
    saver.run()
    
