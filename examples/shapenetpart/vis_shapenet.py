import numpy as np
import os

cat2id = {0: 'airplane', 1: 'bag', 2: 'cap', 3: 'car', 4: 'chair',
                       5: 'earphone', 6: 'guitar', 7: 'knife', 8: 'lamp', 9: 'laptop',
                       10: 'motor', 11: 'mug', 12: 'pistol', 13: 'rocket', 14: 'skateboard', 15: 'table'}

cls2color = {0: [0,255,0], 1: [0,0,255], 2: [255,255,0], 3: [255,255,0],
                        4: [255,0,255], 5: [100,100,255], 
                        6: [200,200,100],7: [170,120,200], 
                        8: [0,255,0], 9: [0,0,255], 10: [255,255,0], 11: [255,255,0],
                        12: [0,255,0], 13: [0,0,255], 14: [255,255,0], 15: [255,255,0],
                        16: [0,255,0], 17: [0,0,255], 18: [255,255,0], 
                        19: [0,255,0], 20: [0,0,255], 21: [255,255,0], 
                        22: [200,200,100],23: [170,120,200], 
                        24: [0,255,0], 25: [0,0,255], 26: [255,255,0], 27: [255,255,0],
                        28: [200,200,100],29: [170,120,200], 
                        30: [0,255,0], 31: [0,0,255], 32: [255,255,0], 33: [255,255,0], 34: [200,200,100], 35: [170,120,200], 
                        36: [200,200,100],37: [170,120,200], 
                        38: [200,200,100],39: [170,120,200], 40: [200,200,100],
                        41: [200,200,100],42: [170,120,200], 43: [200,200,100],
                        44: [200,200,100],45: [170,120,200], 46: [200,200,100],
                        47: [200,200,100],48: [170,120,200], 49: [200,200,100],


}

visual_dir = 'log/shapenetpart/visual/'
def vis_shapenet(data, preds, idx):
    points = data['pos']
    file_names = data['cls']
    gts = data['y']
    #print(file_names)
    batch_size, num_point, _ = points.size()
    for i in range(batch_size):
        point = points[i].cpu().numpy()   
        file_name = visual_dir+cat2id[int(file_names[i][0].cpu().numpy())] + "_" + "{:04d}".format(idx) + "_" +"{:04d}".format(i)
        gt = gts[i].cpu().numpy()
        pred = preds[i].cpu().numpy()
        fout_gt = open(file_name+"_gt.obj", 'w')
        fout_pred = open(file_name+"_pred.obj", 'w')
        for j in range(num_point):
            c_gt = cls2color[gt[j]]
            c_pred = cls2color[pred[j]]
            fout_gt.write('v %f %f %f %f %f %f\n' % (point[j, 0], point[j, 1], point[j, 2], c_gt[0], c_gt[1], c_gt[2]))
            fout_pred.write('v %f %f %f %f %f %f\n' % (point[j, 0], point[j, 1], point[j, 2], c_pred[0], c_pred[1], c_pred[2]))
        fout_gt.close()
        fout_pred.close()
        #print(gt[0], pred[0])

    return 0
    #write_obj(points, colors, out_filename)


def write_obj(points, colors, out_filename):
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        c = colors[i]
        fout.write('v %f %f %f %f %f %f\n' % (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))
    fout.close()

colors = {'ceiling':[0,255,0],
          'floor':[0,0,255],
          'wall':[0,255,255],
          'beam':[255,255,0],
          'column':[255,0,255],
          'window':[100,100,255],
          'door':[200,200,100],
          'table':[170,120,200],
          'chair':[255,0,0],
          'sofa':[200,100,100],
          'bookcase':[10,200,100],
          'board':[200,200,200],
          'clutter':[50,50,50]}