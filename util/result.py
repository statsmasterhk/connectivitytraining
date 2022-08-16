import os, csv
import numpy as np

### Direct copy from util/result.py import_subsequences_results function with slight modification
def import_subsequences_results(path):
    # import trajectories from result
    f = open(path, "r")
    csv_reader = csv.reader(f, delimiter=",")
    detected_boxes = []
    stf = 0
    edf = 0
    sf_flag = 0
    count = 0
    id_set = set()
    for row in csv_reader:
        if count < 3:
            count += 1
            continue
        if sf_flag == 0:
            stf = int(row[0].split("/img")[-1].split(".")[0])
            sf_flag = 1 
        edf = int(row[0].split("/img")[-1].split(".")[0])
        boxes = [ int(r) for r in row[3:] if r != ""]
        bxs = []
        for c in range(int(len(boxes) / 5)):
            idd = int(boxes[c*5])
            bxs.append([int(boxes[c*5+2]), int(boxes[c*5+1]), int(boxes[c*5+2]+boxes[c*5+4]), int(boxes[c*5+1]+boxes[c*5+3]), idd]) # y1, x1, y2, x2, id
            if idd not in id_set:
                id_set.add(idd)
        detected_boxes.append(bxs)

    return stf, edf, detected_boxes