import os
import numpy as np
import pandas as pd
import random as rn
import time
import glob
import shutil
import torch
import argparse
import SimpleITK as sitk
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join,load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

from utils.postprocess import remove_islands_image

def write_image_from_dict(img,props,out_path):
    out_itk_img = sitk.GetImageFromArray(img)
    out_itk_img.SetSpacing(props['spacing'])
    out_itk_img.SetOrigin(props['origin'])
    out_itk_img.SetDirection(props['direction'])
    sitk.WriteImage(out_itk_img, out_path, useCompression=True)

def combine_sub_files(sub_lbl_files, lbl_file, sub_ranges, props):
    ret = np.zeros(props['size'][::-1]).astype(np.uint8)
    print('Combining subfiles...')
    for sub_lbl_file, sub_range in zip(sub_lbl_files, sub_ranges):
        print(sub_lbl_file, sub_range)
        start, end = sub_range
        tmp = sitk.GetArrayFromImage(sitk.ReadImage(sub_lbl_file))
        ret[start:end,...] = np.maximum(tmp, ret[start:end,...])
        # ret[start:end,...] = tmp
    return ret

def generate_sub_files(img_file, limiter, out_dir, model_type='3d'):
    _, ext = os.path.splitext(img_file)
    itk_img = sitk.ReadImage(img_file)
    props = {'spacing': itk_img.GetSpacing(),
             'direction':itk_img.GetDirection(),
             'origin': itk_img.GetOrigin(),
             'size': itk_img.GetSize()}
    npy_img = sitk.GetArrayFromImage(itk_img)
    img_size = itk_img.GetSize()
    print('Processing original image size ', img_size)
    print('Z limiter value: ', limiter)
    sub_ranges = get_ranges(props['size'][2], l=limiter, model_type=model_type)

    sub_img_files, sub_lbl_files = [], []
    print('Generating sub volumes...')
    for i, sub_range in enumerate(sub_ranges):
        start, end = sub_range
        sub_img = npy_img[start:end,...]
        
        sub_img_file, sub_lbl_file =os.path.join(out_dir, os.path.basename(img_file).replace(ext, f'_sub{i}_img{ext}')),\
                                    os.path.join(out_dir, os.path.basename(img_file).replace(ext, f'_sub{i}_label.mhd'))
        sub_img_files.append(sub_img_file), sub_lbl_files.append(sub_lbl_file)
        print('Sub image ', sub_img_file)
        print('Size ', sub_img.shape)
        write_image_from_dict(sub_img, props, sub_img_file)

    return sub_img_files, sub_lbl_files, sub_ranges, props

def get_ranges(sz, l=600, model_type='3d'):
    # p = int((sz % l) / (sz // l))
    p = 200
    # p = sz % l
    if model_type == '3d':
        if sz > l*2:
            ret = [[0, l], *[[x - p, x+l - p] for x in range(l, sz, l)], [sz -l, sz]]
        else:
            ret = [[0, l], [sz -l, sz]]
    else:
        ret = [[x, min(x + l, sz)] for x in range(0, sz, l)]
    return ret

def check_volume_size(img_file, l):
    itk_img = sitk.ReadImage(img_file)
    img_sz = itk_img.GetSize()[-1]
    if img_sz > l:
        print(f'Image size {img_sz} was found larger than limiter {l}')
        return True
    else:
        print(f'Image size {img_sz} was found smaller than limiter {l}')
        return False

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

def infer_from_files(predictor, source_files, target_files, args):
    predictor.predict_from_files(source_files,target_files,
                                save_probabilities=args.save_probabilities, 
                                save_entropy=args.save_entropy, 
                                overwrite=False,
                                num_processes_preprocessing=1, 
                                num_processes_segmentation_export=1,
                                folder_with_segs_from_prev_stage=None, 
                                num_parts=1, 
                                part_id=0)

def set_session(args):
    os.environ["nnUNet_raw"] = args.nnUNet_raw
    os.environ["nnUNet_preprocessed"] = args.nnUNet_preprocessed
    os.environ["nnUNet_results"] = args.nnUNet_results
    os.environ["OMP_NUM_THREADS"] = str(1)
    os.environ["nnUNet_n_proc_DA"] = str(0)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    rn.seed(args.seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predictor of nnUNet')
    parser.add_argument('--dataset_id', type=int, #nargs='+', 
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset") # Defaults to Osaka muscle segmentation fold 1
    parser.add_argument('--model_type',required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3f_fullres. Configurations that do not exist for some dataset will be skipped.')
    parser.add_argument('--fold', default=0)
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--save_entropy', action='store_true')
    parser.add_argument('--save_probabilities', action='store_true')
    parser.add_argument('--nnUNet_raw', type=str, default="/mnt/nnUNet_raw")
    parser.add_argument('--nnUNet_preprocessed', type=str, default="/mnt/nnUNet_preprocessed")
    parser.add_argument('--nnUNet_results', type=str, default="/mnt/nnUNet_results")
    parser.add_argument('--indir', type=str, default=None)    
    parser.add_argument('--outdir', type=str, default=None)    
    parser.add_argument('--datalist', type=str, default=None)    
    parser.add_argument('--tag', type=str, default=None)    
    parser.add_argument('--case_tag', type=str, default=None)    
    parser.add_argument('--trainer', type=str, default='nnUNetTrainer')    
    parser.add_argument('--seed', type=int, default=0)    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limiter', type=int, default=1000)

    args = parser.parse_args()
    print(args)
    set_session(args)

    # nnUNetv2_predict -d 3 -f 0 -c 3d_lowres -i imagesTs -o imagesTs_predlowres --continue_prediction

    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.50,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        # perform_everything_on_gpu=False,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    if args.fold in [str(x) for x in range(5)]:
        checkpoint_name = 'checkpoint_final.pth' \
                        if os.path.exists(join(args.nnUNet_results, f'Dataset{args.dataset_id:03d}_{args.tag}_fold{args.fold}/{args.trainer}__nnUNetPlans__{args.model_type[0]}/fold_{args.fold}/checkpoint_final.pth')) \
                        else 'checkpoint_latest.pth'
        predictor.initialize_from_trained_model_folder(
            join(args.nnUNet_results, f'Dataset{args.dataset_id:03d}_{args.tag}_fold{args.fold}/{args.trainer}__nnUNetPlans__{args.model_type[0]}'),
            use_folds=(args.fold,),
            checkpoint_name=checkpoint_name,
        )
    else:
        checkpoint_name = 'checkpoint_final.pth' \
                        if os.path.exists(join(args.nnUNet_results, f'Dataset{args.dataset_id:03d}_{args.tag}_{args.fold}/{args.trainer}__nnUNetPlans__{args.model_type[0]}/fold_{args.fold}/checkpoint_final.pth')) \
                        else 'checkpoint_latest.pth'
        predictor.initialize_from_trained_model_folder(
            join(args.nnUNet_results, f'Dataset{args.dataset_id:03d}_{args.tag}_{args.fold}/{args.trainer}__nnUNetPlans__{args.model_type[0]}'),
            use_folds=(args.fold,),
            checkpoint_name=checkpoint_name,
        )
#     # variant 1: give input and output folders
#     predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
#                                  join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
#                                  save_probabilities=False, overwrite=False,
#                                  num_processes_preprocessing=2, num_processes_segmentation_export=2,
#                                  folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # variant 2, use list of files as inputs. Note how we use nested lists!!!
    if args.indir is not None:
        indir = args.indir
    else:
        indir = join(args.nnUNet_raw, f'Dataset{args.dataset_id:03d}_{args.tag}_fold{args.fold}/imagesTs')
    
    if args.outdir is not None:
        outdir = args.outdir
    else:
        outdir = join(args.nnUNet_results, f'Dataset{args.dataset_id:03d}_{args.tag}_fold{args.fold}/{args.trainer}__nnUNetPlans__{args.model_type[0]}/predictions')
    os.makedirs(outdir, exist_ok=True)
    if args.datalist is None:
        print('No datalist is loaded')
    elif any([('.json' in args.datalist), ('.txt' in args.datalist)]):
        if '.json' in args.datalist:
            loaded_json = load_json(args.datalist)
            caseid_list = loaded_json[0]['test']
        elif '.txt' in args.datalist:
            caseid_list = read_datalist(args.datalist)
        print(caseid_list)
    else:
        loaded_json= load_json(join(args.nnUNet_raw, f'Dataset{args.dataset_id:03d}_{args.tag}_fold{args.fold}/splits_final.json'))
        caseid_list = loaded_json[0]['test']
        print(caseid_list)
    
    # source_files = [join(indir, f'{x}_0000.nii.gz') for x in caseid_list]
    # target_files = [join(outdir, f'{x}_label.nii.gz') for x in caseid_list]
    # predictor.predict_from_files(source_files,target_files,
    #                                 save_probabilities=True, overwrite=True,
    #                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
    #                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
    #### NII.GZ#####
    # for _case in caseid_list:
    #       if not os.path.exists(join(outdir, f'{_case}/{_case}-muscles_label.mhd')):
    #            source_files.append([join(indir, f'{_case}_000{str(x)}.nii.gz') for x in range(int(args.n_channels))])
    #            target_files.append(join(outdir, f'{_case}/{_case}-muscles_label.nii.gz'))
    #            print(source_files)
    #            print(target_files)
    #            predictor.predict_from_files(source_files,target_files,
    #                                          save_probabilities=False, 
    #                                          overwrite=False,
    #                                          num_processes_preprocessing=4, 
    #                                          num_processes_segmentation_export=8,
    #                                          folder_with_segs_from_prev_stage=None, 
    #                                          num_parts=1, 
    #                                          part_id=0)
    #            source_files, target_files = [], []
    source_files, target_files = [], []
    if args.recursive:
        image_list = [x for x in glob.glob(join(indir, '**/*.mhd'), recursive=True)]
        out_labels_list = []
        for x in image_list:
            img_fname = os.path.basename(os.path.splitext(x)[0])
            case_name = os.path.basename(os.path.dirname(x))
            out_label_path = join(outdir, img_fname, img_fname+'-org_muscles_label.mhd')
            out_labels_list.append(out_label_path)
    else:
        image_list = []
        for case in caseid_list:
            case_id = case.split('_')[0]
            image_list.extend([join(indir, case_id, f'{case}.mhd')])
        out_labels_list = []
        for x in image_list:
            img_fname = os.path.basename(os.path.splitext(x)[0])
            case_name = os.path.basename(os.path.dirname(x))
            out_label_path = join(outdir, case_name, img_fname+'-org_muscles_label.mhd')
            out_labels_list.append(out_label_path)
    print('Image files: ', *image_list, sep='\n')
    print('Out label files: ', *out_labels_list, sep='\n')
    
    for img_file, lbl_file in zip(image_list, out_labels_list):
        os.makedirs(os.path.dirname(lbl_file), exist_ok=True)
        if not os.path.exists(lbl_file):
            try:
                st = time.time()
                print('Started Inference...')
                print('Image file ',  img_file)
                print('Label file ',  lbl_file)
                # Check volume size
                # If the volume > limiter --> go to partitioner --> generate source files + target lists --> for each source/target segmet --> combine
                # Else 
                vol_flag = check_volume_size(img_file, args.limiter)
                if vol_flag:
                    tmp_out_dir = join(outdir,f'tmp_{os.path.splitext(os.path.basename(img_file))[0]}')
                    os.makedirs(tmp_out_dir, exist_ok=True)
                    sub_img_files, sub_lbl_files, sub_ranges, props = generate_sub_files(img_file, args.limiter, tmp_out_dir, model_type="3d" if ('3d' in args.model_type[0]) else "2d")
                    for sub_img_file, sub_lbl_file in zip(sub_img_files, sub_lbl_files):
                        if not os.path.exists(sub_lbl_file):
                            infer_from_files(predictor, [[sub_img_file]], [sub_lbl_file], args)
                    ret = combine_sub_files(sub_lbl_files, lbl_file, sub_ranges,props)
                    write_image_from_dict(ret, props, lbl_file)
                    shutil.rmtree(tmp_out_dir)
                else:
                    infer_from_files(predictor, [[img_file]], [lbl_file], args)
                end = time.time()
                print('Inference time: %0.3f sec' % (end - st))
                if args.postprocess:
                    st_post = time.time()
                    remove_islands_image(lbl_file)
                    end_post = time.time()
                    print('Postprocessing time: %0.3f sec' % (end_post - st_post))
                # source_files, target_files = [], []
                end = time.time()
                print('Total inference time: %0.3f sec' % (end - st))
            except Exception as e:
                with open(join(outdir, 'unprocessed.txt'), 'a') as w:
                    w.writelines(lbl_file + '\n')
                    # w.writelines(e + '\n')
        else:
            print(f'Image file {img_file} already segmented...', )

    #### MHD for UKBiobank ####
    # for _case in caseid_list:
    #         case_out_dir = join(outdir, _case)
    #         os.makedirs(case_out_dir, exist_ok=True)
    #         if not os.path.exists(join(case_out_dir, f'{_case}_label.mhd')):
    #             source_files.append([join(indir, f'{_case}', f'{_case}_{str(x)}.mhd') for x in ['in', 'opp', 'F', 'W']])
    #             target_files.append(join(case_out_dir, f'{_case}_label'))
    #             print([os.path.exists(x) for x in source_files[0]])
    #             print(all([os.path.exists(x) for x in source_files[0]]))
    #             print(source_files)
    #             if all([os.path.exists(x) for x in source_files[0]]):
    #                     try:
    #                         predictor.predict_from_files(source_files,target_files,
    #                                                         save_probabilities=False, 
    #                                                         overwrite=False,
    #                                                         num_processes_preprocessing=4, 
    #                                                         num_processes_segmentation_export=4,
    #                                                         folder_with_segs_from_prev_stage=None, 
    #                                                         num_parts=1, 
    #                                                         part_id=0)
    #                         source_files, target_files = [], []
    #                     except:
    #                         with open(join(outdir, 'unprocessed_log.txt'), 'a') as f:
    #                             f.writelines(f'{_case}\n')
    #                         source_files, target_files = [], []
    #             else:
    #                 with open(join(outdir, 'unprocessed_log.txt'), 'a') as f:
    #                     f.writelines(f'{_case}\n')
    #                 source_files, target_files = [], []
                       

#     # variant 2.5, returns segmentations
#     indir = join(nnUNet_raw, 'Dataset003_Liver/imagesTs')
#     predicted_segmentations = predictor.predict_from_files([[join(indir, 'liver_152_0000.nii.gz')],
#                                                             [join(indir, 'liver_142_0000.nii.gz')]],
#                                                            None,
#                                                            save_probabilities=True, overwrite=True,
#                                                            num_processes_preprocessing=2,
#                                                            num_processes_segmentation_export=2,
#                                                            folder_with_segs_from_prev_stage=None, num_parts=1,
#                                                            part_id=0)

#     # predict several npy images
#     from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

#     img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_147_0000.nii.gz')])
#     img2, props2 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_146_0000.nii.gz')])
#     img3, props3 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_145_0000.nii.gz')])
#     img4, props4 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_144_0000.nii.gz')])
#     # we do not set output files so that the segmentations will be returned. You can of course also specify output
#     # files instead (no return value on that case)
#     ret = predictor.predict_from_list_of_npy_arrays([img, img2, img3, img4],
#                                                     None,
#                                                     [props, props2, props3, props4],
#                                                     None, 2, save_probabilities=False,
#                                                     num_processes_segmentation_export=2)

#     # predict a single numpy array
#     img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_147_0000.nii.gz')])
#     ret = predictor.predict_single_npy_array(img, props, None, None, True)

#     # custom iterator

#     img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_147_0000.nii.gz')])
#     img2, props2 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_146_0000.nii.gz')])
#     img3, props3 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_145_0000.nii.gz')])
#     img4, props4 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_144_0000.nii.gz')])


#     # each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
#     # If 'ofile' is None, the result will be returned instead of written to a file
#     # the iterator is responsible for performing the correct preprocessing!
#     # note how the iterator here does not use multiprocessing -> preprocessing will be done in the main thread!
#     # take a look at the default iterators for predict_from_files and predict_from_list_of_npy_arrays
#     # (they both use predictor.predict_from_data_iterator) for inspiration!
#     def my_iterator(list_of_input_arrs, list_of_input_props):
#         preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
#         for a, p in zip(list_of_input_arrs, list_of_input_props):
#             data, seg = preprocessor.run_case_npy(a,
#                                                   None,
#                                                   p,
#                                                   predictor.plans_manager,
#                                                   predictor.configuration_manager,
#                                                   predictor.dataset_json)
#             yield {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properites': p, 'ofile': None}


#     ret = predictor.predict_from_data_iterator(my_iterator([img, img2, img3, img4], [props, props2, props3, props4]),
#                                                save_probabilities=False, num_processes_segmentation_export=3)
