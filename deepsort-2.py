print("=====  默认采用 动态 跳帧模型，如果上一次检测的的框，和本次检测的框，存在有框的交并比低于某个阈值，跳帧级别下降，反之，则上升；无补充回溯检测 =====")

import torch
import torchvision
import numpy as np
import cv2
import motmetrics
import scipy
import lap
import motmetrics as mm
import pandas as pd
import os
import sys
from pathlib import Path
from torchvision.ops import box_iou 
sys.path.append(r'D:\deepsort\deep_sort_pytorch-master')

# 测试 deep_sort_pytorch
try:
    from deep_sort.deep_sort import DeepSort 
    print("deep_sort_pytorch OK")
except Exception as e:
    print("deep_sort_pytorch 导入失败:", e)

print(torch.cuda.is_available())

level=1
cnt=0

def get_v(level: int) -> int:
    """
    输入 LEVEL（1~4），返回对应的 V。
      LEVEL 1 → 15
      LEVEL 2 → 10
      LEVEL 3 →  5
      LEVEL 4 →  3
    其余 LEVEL 会抛出 ValueError。
    """
    mapping = {1: 16, 2: 8, 3: 4, 4: 2}
    try:
        return mapping[level]
    except KeyError:
        raise ValueError(f"无效的 LEVEL: {level}，应为 1~4 之一")


# ──────────────────────────────────────────────
# 1. 统一 BOX 格式
# ──────────────────────────────────────────────
def to_xyxy(boxes_xywh):
    """
    (N,4) of [x1,y1,w,h]  → (N,4) of [x1,y1,x2,y2]
    """
    xyxy = boxes_xywh.copy()
    xyxy[:, 2] = xyxy[:, 0] + xyxy[:, 2]   # x2 = x1 + w
    xyxy[:, 3] = xyxy[:, 1] + xyxy[:, 3]   # y2 = y1 + h
    return xyxy
# ──────────────────────────────────────────────
# 2. 读取 MOT16 det.txt
# ──────────────────────────────────────────────
def read_mot_det_results(det_path, conf_thres: float = 0):
    """
    读取 MOT16 det.txt，返回 dict:
    {frame_id: ndarray(N, 5)  # (x1,y1,w,h,score)}
    """
    det_path = Path(det_path)
    frames = {}
    with det_path.open('r') as f:
        for line in f:
            f = line.strip().split(',')
            frame, x1, y1, w, h, score = (int(f[0]), *map(float, f[2:6]), float(f[6]))
            if score < conf_thres:
                continue
            frames.setdefault(frame, []).append([x1, y1, w, h, score])
    # 转成 ndarray，便于后续运算
    for k in frames:
        frames[k] = np.asarray(frames[k], dtype=np.float32)
    return frames



# 3. 地址列表

root = r"D:\deepsort\MOT16\train\MOT16-"

for i in range(1, 15):   # range(1, 15) 产生 1~14
    # 2. 初始化 DeepSort

    s = f"{i:02d}"   

    rootDir = root+s
    if not os.path.exists(rootDir):
        continue
    print(rootDir)
    deepsort = DeepSort(
        model_path="deep_sort_pytorch-master/deep_sort/deep/checkpoint/ckpt.t7",max_age=70,min_confidence=0,n_init=1, use_cuda=False
    )

    det_txt = rootDir+r"\gt\gt.txt"
    gt_txt = rootDir+r"\gt\gt.txt"
    res_txt = "L_"+str(level)+"mot16_deepsort.txt"
    seq_img_dir = Path(rootDir+r"\img1")  # ← 对应帧图像目录
    frame_dict = read_mot_det_results(det_txt)
    # 1. 文件路径
    gt_file  = Path(gt_txt)
    res_file = Path(res_txt)   # 上一步生成的结果文件
    results = []  # 存储跟踪结果：frame, track_id, x1, y1, w, h, score



    
    det2 = np.empty((0, 4), dtype=np.float32)
    for frame_id in sorted(frame_dict):
        #print(frame_id)
        dets = frame_dict[frame_id]          # (N,5)  x1 y1 w h score
         # 读取当前帧图像（如果只做 IOU 跟踪，可传 None）
        #print(frame_dict[frame_id] )
        img_path = seq_img_dir / f"{frame_id:06d}.jpg"
        img = cv2.imread(str(img_path))           # BGR
        if img is None:
            raise FileNotFoundError(img_path)

        
        v=get_v(level)
        if frame_id%v == 0 and len(dets) != 0:

            # 1) bbox_xywh 需为 (cx,cy,w,h)
            bbox_xywh = dets[:, :4].copy()
            bbox_xywh[:, 0] += bbox_xywh[:, 2] / 2.0     # x1→cx
            bbox_xywh[:, 1] += bbox_xywh[:, 3] / 2.0     # y1→cy
            scores = dets[:, 4]
            clss = np.zeros_like(scores, dtype=np.int32)   # MOT16 行人 → 类别 0
            #print("update")

            
            det1   = to_xyxy(dets[:, :4])
            #print(det2)
            #print(det1)
            iou_mat = box_iou(torch.from_numpy(det1),torch.from_numpy(det2)).numpy()   # (K,N)
            if not iou_mat.size == 0: 
                max_iou_per_track = iou_mat.max(axis=1)
                K = len(max_iou_per_track)
                k = int(np.ceil(0.95 * K))          # 前 90 %，向上取整，最少 1
                # ① 先整体降序排序
                sorted_desc = np.sort(max_iou_per_track)[::-1]   # 从大到小

                # ② 取最大的前 k 个
                top_90 = sorted_desc[:k]           # 长度 = k

                # ③ 在这 k 个里求最小值
                minIou = top_90.min()
                #print(minIou)
                if minIou > 0.5:
                    if level>1:
                        level=level-1
                        #print("level->"+str(level))
                if minIou < 0.2:
                    if level<4:
                        level=level+2
                        if level> 4:
                            level=4
                        #print("level->"+str(level))
            det2=det1
            cnt = cnt+1
            
        else:
            bbox_xywh = np.empty((0, 4), dtype=np.float32)
            scores     = np.empty((0,), dtype=np.float32)
            clss       = np.empty((0,), dtype=np.int32)
            #print("predict")
        
        #print(bbox_xywh)
        outputs,_ = deepsort.update(bbox_xywh, scores,clss, img)  # ← 传入 img 以提取外观特征
        
        #print("output")
        #print(outputs)
        #print(str(len(dets))+"  "+str(len(outputs))+"  "+str(len(bbox_xywh)))
        # DeepSort 返回: [x1,y1,x2,y2,track_id]
        
        for x1, y1, x2, y2, clss_id,track_id in outputs:
            w, h = x2 - x1, y2 - y1
            results.append([frame_id, track_id, x1, y1, w, h, 1, -1, -1, -1])

    print(f"[INFO] Finished. Total tracks: {len(results)}")


    # 4. 保存结果为 MOT 格式
    with open(res_txt, "w") as f:
        for r in results:
            f.write(','.join([str(int(r[0])), str(int(r[1]))] + ['%.2f' % q for q in r[2:]]) + "\n")





    # 2. 载入（fmt = "mot15-2DBox" 兼容所有 10 列格式）
    gt  = mm.io.loadtxt(gt_file,  fmt="mot16", min_confidence=1)   # 只保留行人(类别=1)
    res = mm.io.loadtxt(res_file, fmt="mot16")

    print(gt.groupby(['ClassId','Confidence']).size())
    print(f"列名: {gt.columns.tolist()}")
    print(res.groupby(['ClassId','Confidence']).size())
    print(f"列名: {res.columns.tolist()}")

    # 3. 创建一个 Accumulator 并做逐帧匹配
    #    dissimilarity = 'iou' 表示用 IoU 匹配；阈值给 0.5
    acc = mm.utils.compare_to_groundtruth(
        gt, res,
        dist='iou',           # 使用 dist 而不是 dist_func
        distth=0.5           # 使用 distth 而不是 dist_thres
    )


    # 4. 统计指标
    mh = mm.metrics.create()
    metrics   = mm.metrics.motchallenge_metrics          # = ['num_frames','mota','motp',...]
    summary   = mh.compute(acc, metrics=metrics, name='DeepSort')

    # 5. 打印
    strsummary = mm.io.render_summary(
                    summary,
                    formatters=mh.formatters,             # 保留 motmetrics 默认格式化
                    namemap=mm.io.motchallenge_metric_names)  # 正确的路径

    print(strsummary)
    print("total frame : "+str(cnt))
    cnt=0

print("Done! 跟踪结果已保存到 mot16_deepsort.txt")
