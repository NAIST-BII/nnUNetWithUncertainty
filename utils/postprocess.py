import numpy as np
import cc3d
from skimage.measure import regionprops 
import mhd
import time
import glob
from .io import read_image, write_image
import tqdm
area_th = 0.05 


def remove_islands(lbl_img):
    conn_lbl_img, N = cc3d.connected_components(lbl_img, return_N=True)
    props = regionprops(conn_lbl_img)
    out_img = np.zeros_like(conn_lbl_img)
    max_area = np.max([prop.area for prop  in props])
    
    if area_th >= 1:
        th = area_th
    else:
        th = int(area_th*max_area)

    for i, prop in enumerate(props):
        area = prop.area
        if area > th:
            out_img[conn_lbl_img == (i+1)]= 1
    
    return out_img

def remove_islands_image(path):
    print('Postprocessing ', path)
    lbl_img, es, offset = read_image(path)

    start = time.time()

    out_lbl_img = np.zeros_like(lbl_img)

    vals = np.setdiff1d(np.unique(lbl_img), [0])

    for val in vals:
        tmp =  lbl_img == val
        tmp = remove_islands(tmp)
        out_lbl_img = np.where(tmp==1, val, out_lbl_img)


    end = time.time()
    total_time = end - start

    out_path = path.replace('org_', '')
    mhd.write(out_path, 
            out_lbl_img, 
            header={
                'ElementSpacing':es,
                'CompressedData':True,
                'Offset': offset
            })
    
    print(f"Done..\n {out_path}")
    print(f"Computation time: {total_time:0.3f} secs")