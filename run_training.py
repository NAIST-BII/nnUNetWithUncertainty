import os
import numpy as np
import random as rn
import torch
import argparse

def set_session(args):
    os.environ["nnUNet_raw"] = args.nnUNet_raw
    os.environ["nnUNet_preprocessed"] = args.nnUNet_preprocessed
    os.environ["nnUNet_results"] = args.nnUNet_results
    os.environ["OMP_NUM_THREADS"] = str(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    rn.seed(args.seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trainer of nnUNet')
    parser.add_argument('--dataset_id', nargs='+', type=int,
                        help="[REQUIRED] List of dataset IDs. Example: 2 4 5. This will run fingerprint extraction, experiment "
                             "planning and preprocessing for these datasets. Can of course also be just one dataset") # Defaults to Osaka muscle segmentation fold 1
    parser.add_argument('--model_type',required=False, default=['2d', '3d_fullres', '3d_lowres'], nargs='+',
                        help='[OPTIONAL] Configurations for which the preprocessing should be run. Default: 2d 3f_fullres '
                             '3d_lowres. 3d_cascade_fullres does not need to be specified because it uses the data '
                             'from 3f_fullres. Configurations that do not exist for some dataset will be skipped.')
    parser.add_argument('--fold', type=str, default="0")
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--nnUNet_raw', type=str, default="/mnt/nnUNet_raw")
    parser.add_argument('--nnUNet_preprocessed', type=str, default="/mnt/nnUNet_preprocessed")
    parser.add_argument('--nnUNet_results', type=str, default="/mnt/nnUNet_results")
    parser.add_argument('--seed', type=int, default=0)    
    parser.add_argument('--gpu', type=int, default=0)

    # Preparation arguments
    parser.add_argument('-fpe', type=str, required=False, default='DatasetFingerprintExtractor',
                        help='[OPTIONAL] Name of the Dataset Fingerprint Extractor class that should be used. Default is '
                             '\'DatasetFingerprintExtractor\'.')
    parser.add_argument('-npfp', type=int, default=8, required=False,
                        help='[OPTIONAL] Number of processes used for fingerprint extraction. Default: 8')
    parser.add_argument("--verify_dataset_integrity", required=False, default=False, action="store_true",
                        help="[RECOMMENDED] set this flag to check the dataset integrity. This is useful and should be done once for "
                             "each dataset!")
    parser.add_argument('--no_pp', default=False, action='store_true', required=False,
                        help='[OPTIONAL] Set this to only run fingerprint extraction and experiment planning (no '
                             'preprocesing). Useful for debugging.')
    parser.add_argument("--clean", required=False, default=False, action="store_true",
                        help='[OPTIONAL] Set this flag to overwrite existing fingerprints. If this flag is not set and a '
                             'fingerprint already exists, the fingerprint extractor will not run. REQUIRED IF YOU '
                             'CHANGE THE DATASET FINGERPRINT EXTRACTOR OR MAKE CHANGES TO THE DATASET!')
    parser.add_argument('-pl', type=str, default='ExperimentPlanner', required=False,
                        help='[OPTIONAL] Name of the Experiment Planner class that should be used. Default is '
                             '\'ExperimentPlanner\'. Note: There is no longer a distinction between 2d and 3d planner. '
                             'It\'s an all in one solution now. Wuch. Such amazing.')
    parser.add_argument('-gpu_memory_target', default=8, type=int, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom GPU memory target. Default: 8 [GB]. Changing this will '
                             'affect patch and batch size and will '
                             'definitely affect your models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')
    parser.add_argument('-preprocessor_name', default='DefaultPreprocessor', type=str, required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom preprocessor class. This class must be located in '
                             'nnunetv2.preprocessing. Default: \'DefaultPreprocessor\'. Changing this may affect your '
                             'models performance! Only use this if you really know what you '
                             'are doing and NEVER use this without running the default nnU-Net first (as a baseline).')
    parser.add_argument('-overwrite_target_spacing', default=None, nargs='+', required=False,
                        help='[OPTIONAL] DANGER ZONE! Sets a custom target spacing for the 3d_fullres and 3d_cascade_fullres '
                             'configurations. Default: None [no changes]. Changing this will affect image size and '
                             'potentially patch and batch '
                             'size. This will definitely affect your models performance! Only use this if you really '
                             'know what you are doing and NEVER use this without running the default nnU-Net first '
                             '(as a baseline). Changing the target spacing for the other configurations is currently '
                             'not implemented. New target spacing must be a list of three numbers!')
    parser.add_argument('-overwrite_plans_name', default='nnUNetPlans', required=False,
                        help='[OPTIONAL] uSE A CUSTOM PLANS IDENTIFIER. If you used -gpu_memory_target, '
                             '-preprocessor_name or '
                             '-overwrite_target_spacing it is best practice to use -overwrite_plans_name to generate a '
                             'differently named plans file such that the nnunet default plans are not '
                             'overwritten. You will then need to specify your custom plans file with -p whenever '
                             'running other nnunet commands (training, inference etc)')
    parser.add_argument('-np', type=int, nargs='+', default=None, required=False,
                        help="[OPTIONAL] Use this to define how many processes are to be used. If this is just one number then "
                             "this number of processes is used for all configurations specified with -c. If it's a "
                             "list of numbers this list must have as many elements as there are configurations. We "
                             "then iterate over zip(configs, num_processes) to determine then umber of processes "
                             "used for each configuration. More processes is always faster (up to the number of "
                             "threads your PC can support, so 8 for a 4 core CPU with hyperthreading. If you don't "
                             "know what that is then dont touch it, or at least don't increase it!). DANGER: More "
                             "often than not the number of processes that can be used is limited by the amount of "
                             "RAM available. Image resampling takes up a lot of RAM. MONITOR RAM USAGE AND "
                             "DECREASE -np IF YOUR RAM FILLS UP TOO MUCH!. Default: 8 processes for 2d, 4 "
                             "for 3d_fullres, 8 for 3d_lowres and 4 for everything else")
    parser.add_argument('--verbose', required=False, action='store_true',
                        help='Set this to print a lot of stuff. Useful for debugging. Will disable progrewss bar! '
                             'Recommended for cluster environments')

    # Training
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='[OPTIONAL] Use this flag to specify a custom trainer. Default: nnUNetTrainer')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='[OPTIONAL] Use this flag to specify a custom plans identifier. Default: nnUNetPlans')
    parser.add_argument('-pretrained_weights', type=str, required=False, default=None,
                        help='[OPTIONAL] path to nnU-Net checkpoint file to be used as pretrained model. Will only '
                             'be used when actually training. Beta. Use with caution.')
    parser.add_argument('-num_gpus', type=int, default=1, required=False,
                        help='Specify the number of GPUs to use for training')
    parser.add_argument("--use_compressed", default=False, action="store_true", required=False,
                        help="[OPTIONAL] If you set this flag the training cases will not be decompressed. Reading compressed "
                             "data is much more CPU and (potentially) RAM intensive and should only be used if you "
                             "know what you are doing")
    parser.add_argument('--npz', action='store_true', required=False,
                        help='[OPTIONAL] Save softmax predictions from final validation as npz files (in addition to predicted '
                             'segmentations). Needed for finding the best ensemble.')
    parser.add_argument('--c', action='store_true', required=False,
                        help='[OPTIONAL] Continue training from latest checkpoint')
    parser.add_argument('--val', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to only run the validation. Requires training to have finished.')
    parser.add_argument('--disable_checkpointing', action='store_true', required=False,
                        help='[OPTIONAL] Set this flag to disable checkpointing. Ideal for testing things out and '
                             'you dont want to flood your hard drive with checkpoints.')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                    help="Use this to set the device the training should run with. Available options are 'cuda' "
                         "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                         "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_train [...] instead!")


    args = parser.parse_args()
    set_session(args)

    if args.prepare:
        from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess

           # fingerprint extraction
        print("Fingerprint extraction...")
        extract_fingerprints(args.dataset_id, 
                             args.fpe, 
                             args.npfp, 
                             args.verify_dataset_integrity, 
                             args.clean, 
                             args.verbose)

        # experiment planning
        print('Experiment planning...')
        plan_experiments(args.dataset_id, 
                         args.pl, 
                         args.gpu_memory_target, 
                         args.preprocessor_name, 
                         args.overwrite_target_spacing, 
                         args.overwrite_plans_name)

        # manage default np
        if args.np is None:
            default_np = {"2d": 8, "3d_fullres": 8, "3d_lowres": 8}
            np = [default_np[c] if c in default_np.keys() else 4 for c in args.model_type]
        else:
            np = args.np
        # preprocessing
        if not args.no_pp:
            print('Preprocessing...')
            preprocess(args.dataset_id, 
                       args.overwrite_plans_name, 
                       args.model_type, 
                       np, 
                       args.verbose)


    # Start training
    from nnunetv2.run.run_training import run_training
    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')
    print(device)
    print(args.dataset_id)
    run_training(str(args.dataset_id[0]) if isinstance(args.dataset_id,list) else str(args.dataset_id), 
                 args.model_type[0], 
                 args.fold, 
               #   args.tr, 
                 args.p, 
                 args.pretrained_weights,
                 args.num_gpus, 
                 args.use_compressed,
                 args.npz, 
                 args.c, 
                 args.val, 
                 args.disable_checkpointing,
                 device=device)
