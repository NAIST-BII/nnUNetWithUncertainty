import os
import pandas as pd
import numpy as np
import json
import glob
import mhd
from .medpy.io.load import load

def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

def combine_csvs(root, tag, transpose=False, header =0):
    '''
    Combines a set of CSV files into a single file.
    '''
    out_dir = os.path.join(root, f'*{tag}')
    # print(out_dir)
    files = sorted(glob.glob(out_dir))
    csvs = []
    for file in files:
        _tmp_df = pd.read_csv(file, header=header, usecols=[0,1,2], index_col=0)
        # print(_tmp_df)
        csvs.append(_tmp_df)
    if transpose:
        csvs = pd.concat(csvs, axis=1).T
    else:
        csvs = pd.concat(csvs, axis=0)
    return csvs



def write_datalist(fpath, l):
    '''
    Writes a list to a file at the file path 
    '''
    with open(fpath, 'w') as fw:
        for x in l:
            fw.write('%s\n' % x)

def read_datalist(fpath, field='fileID'):
    '''
    Reads a file path and a field and return corresponding datalist from file
    '''
    datalist = []
    if fpath.endswith('.txt'):
        with open(fpath, 'r') as fr:
            datalist = fr.readlines()
        datalist = [dat.replace('\n', '') for dat in datalist]
    elif fpath.endswith('.csv'):
        df = pd.read_csv(fpath)
        print('Dataframe: ', df)
        datalist = df[field].values.tolist()
    return datalist

        
def load_dict_json_file(file_path):
    '''
    Loads a dictionary from json file
    '''
    try:
        if isinstance(file_path, str):
            assert file_path.endswith('.json'), 'Input should be a JSON file!'
            with open(file_path, 'r') as fp:
                parsed_json = json.load(fp)
            return parsed_json
    except:
        print('Error in JSON file reading')


def read_image(pth, hdr = None, crop=None, output_trans_matrix=False, out_hdr=False):
    '''
    Reads an image path and returns an image with header info
    '''
    _, ext = os.path.splitext( os.path.basename(pth) )
    offset = np.zeros((3,))
    trans_matrix = '1 0 0 0 1 0 0 0 1'
    if ext in ('.mha', '.mhd'):
        [img, img_header] = mhd.read(pth)
        spacing = img_header['ElementSpacing']
        # img.flags.writeable = True
        if 'Offset' in img_header.keys():
            offset = img_header['Offset']
        if 'TransformMatrix' in img_header.keys():
            trans_matrix = img_header['TransformMatrix']
        
    elif ext in ('.nii.gz', '.nii', '.gz'):
        [img, img_header] = load(pth)
        spacing = img_header.get_voxel_spacing()
        img = np.transpose(img, (2,1,0))
        offset = img_header.get_offset()
        for i,d in enumerate(img_header.get_direction().diagonal()):
            if d == -1:
                img = flip(img, i)
    else:
        raise NotImplementedError()
    
    if crop:
        _min, _max = crop
        if _max <= 1:
            _max = int(_max*img.shape[0])
        if 0 <_min < 1:
            _min = int(_min*img.shape[0])
        img = img[_min:_max,...]
        offset = offset + np.array([0,0,_min*spacing[2]])

    if out_hdr:
        img_header['ElementSpacing'] = spacing
        img_header['TransformMatrix'] = trans_matrix
        img_header['Offset'] = offset
        return img, img_header
    elif output_trans_matrix:
        return img, spacing, offset, trans_matrix
    else:
        return img, spacing, offset


def write_image(pth, img, hdr = None):
    '''
    Reads an image path and returns an image with header (if exists)
    '''
    _, ext = os.path.splitext(pth)
    try:
        if (ext == '.mhd') or (ext == '.mha'):
            _es = hdr['ElementSpacing']
            _offset = hdr['Offset'] if 'Offset' in hdr.keys() else [0,0,0]
            mhd.write(pth, img, header={'Offset': _offset,
                                        'ElementSpacing': _es,
                                        'CompressedData': True})
    except Exception as exc:
        print(exc)


def check_dfs_sanity(df1, df2):
    out = df1.index.compare(df2.index)
    if out is not None:
        raise ValueError
    else:
        pass

def read_json (fpath):
    with open(fpath) as f:
        out_dict = json.load(f)
    return out_dict

