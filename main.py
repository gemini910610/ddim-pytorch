import argparse
import numpy
import os
import shutil
import torch
import yaml

# disable scientific notation
torch.set_printoptions(sci_mode=False)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data')
    parser.add_argument('--doc', type=str, required=True, help='A string for documentation purpose. Will be the name of the log folder')
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical', choices=['info', 'debug', 'warning', 'critical'])
    parser.add_argument('--test', action='store_true', help='Whether to test the model')
    parser.add_argument('--sample', action='store_true', help='Whether to produce samples from the model')
    parser.add_argument('--fid', action='store_true')
    parser.add_argument('--interpolation', action='store_true')
    parser.add_argument('--resume_training', action='store_true', help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help='The folder name of samples')
    parser.add_argument('--ni', action='store_true', help='No interaction. Suitable for Slurm Job launcher')
    parser.add_argument('--use_pretrained', action='store_true')
    parser.add_argument('--sample_type', type=str, default='generalized', help='Sampling approach (generalized or ddpm_noisy)', choices=['generalized', 'ddpm_noisy'])
    parser.add_argument('--skip_type', type=str, default='uniform', help='Skip according to (uniform or quadratic)', choices=['uniform', 'quadratic'])
    parser.add_argument('--timesteps', type=int, default=1000, help='Number of steps involved')
    parser.add_argument('--eta', type=float, default=0.0, help='Eta used to control the variances of sigma')
    parser.add_argument('--sequence', action='store_true')

    args = parser.parse_args()
    return args

def load_config(config_file):
    if not os.path.exists(config_file):
        print(f'Configuration file "{config_file}" not found.')
        exit()
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = dict2namespace(config)
    return config

def overwrite_folder(args):
    training = not args.test and not args.sample
    if training:
        overwrite_training_folder(args, config)
    elif args.sample:
        sample_path = os.path.join(args.exp, 'image_samples')
        os.makedirs(sample_path, exist_ok=True)
        overwrite_sampling_folder(args)

def check_overwrite(path, no_interaction):
    if no_interaction:
        return True
    
    response = input(f'Folder "{path}" already exists. Overwrite? (Y/N) ')
    if response.upper() == 'Y':
        return True
    
    return False

def remove_overwrite_folder(path, no_interaction):
    overwrite = check_overwrite(path, no_interaction)
    if not overwrite:
        print(f'Folder "{path}" already exists. Program halted.')
        exit()
    shutil.rmtree(path)

def overwrite_sampling_folder(args):
    image_path = os.path.join(args.exp, 'image_samples', args.image_folder)
    if os.path.exists(image_path):
        if args.fid or args.interpolation:
            return
        
        remove_overwrite_folder(image_path, args.ni)

    os.makedirs(image_path)

def overwrite_training_folder(args, config):
    if args.resume_training:
        return

    log_path = os.path.join(args.exp, 'logs', args.doc)
    if os.path.exists(log_path):
        remove_overwrite_folder(log_path, args.ni)

    os.makedirs(log_path)
    config_file = os.path.join(log_path, 'config.yaml')
    with open(config_file, 'w') as file:
        yaml.dump(config, file)

def dict2namespace(config: dict) -> argparse.Namespace:
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            value = dict2namespace(value)
        setattr(namespace, key, value)
    return namespace

def set_random_seed(seed):
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)
    
    overwrite_folder(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f'Using device: {device}')

    set_random_seed(args.seed)

    log_path = os.path.join(args.exp, 'logs', args.doc)
    print(f'Writing log file to {log_path}')
    print(f'Exp comment = {args.comment}')
