from argparse import Namespace
import os
import sys
import pprint
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import torchvision.transforms.functional as F
import time
import torch.nn as nn
import copy
import shutil
import collections

# from criteria.aging_loss import AgingLoss
from criteria.aging_loss_fpage import AgingLoss
from criteria import id_loss

# sys.path.append(".")
# sys.path.append("..")
sys.path.append("/playpen-nas-ssd/luchao/projects/SAM")

from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp
import dlib
from scripts.align_all_parallel import align_face
from tqdm import tqdm
# import gpytorch
# import matplotlib.pyplot as plt


def run_alignment(image_path):
    predictor = dlib.shape_predictor("/playpen-nas-ssd/luchao/pretrained_weights/dlib/shape_predictor_68_face_landmarks.dat")
    aligned_image = align_face(filepath=image_path, predictor=predictor) 
    # print("Aligned image has shape: {}".format(aligned_image.size))
    return aligned_image 

# def run_on_batch(inputs, net):
#     result_batch = net(inputs.to("cuda").float(), randomize_noise=False, resize=False)
#     return result_batch

def run_on_batch(inputs, net):
    result_batch, result_style = net(inputs.to("cuda").float(), randomize_noise=False, resize=False, return_latents=True)
    return result_batch, result_style

def run_on_batch_blender(inputs, net, net_global, target_ages):
    # this function is same as `perform_forward_pass_blender` in `coach_aging_delta.py`
    _, latent_local = net.forward(inputs.to("cuda").float(), return_latents=True)
    _, latent_global = net_global.forward(inputs.to("cuda").float(), return_latents=True)
    latent_blended = net.blender(latent_local, latent_global, target_ages=target_ages)
    result_batch, _ = net.decoder(
        [latent_blended], 
        input_is_latent=True, 
        randomize_noise=False
        )
    return result_batch, latent_blended
    
# def run_on_batch_blender(inputs, net, net_global):
#     _, latent_local = net.forward(inputs.to("cuda").float(), return_latents=True)
#     _, latent_global = net_global.forward(inputs.to("cuda").float(), return_latents=True)
#     target_ages = inputs[:, -1, 0, 0]
#     # print('sanity check from run_on_batch_blender: ', target_ages)
#     # print('this should align with the target ages')
#     latent_blended = net.blender(latent_local, latent_global, target_ages)
#     result_batch, _ = net.decoder(
#         [latent_blended], 
#         input_is_latent=True, 
#         randomize_noise=False
#         )
#     return result_batch, latent_blended

# def run_on_batch_blender(inputs, net, net_global):
#     # gaussian process
#     _, latent_local = net.forward(inputs.to("cuda").float(), return_latents=True)
#     _, latent_global = net_global.forward(inputs.to("cuda").float(), return_latents=True)
#     target_ages = inputs[:, -1, 0, 0]
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         theta_pred = net.blender.likelihood(net.blender(torch.tensor(target_ages).float().to('cuda')))
#     theta = theta_pred.mean
#     # print('theta_pred: ', theta_pred)
#     # print('theta: ', theta)
#     latent_blended = latent_local * theta + latent_global * (1 - theta)
#     # latent_blended = net.blender(latent_local, latent_global, target_ages)
#     result_batch, _ = net.decoder(
#         [latent_blended], 
#         input_is_latent=True, 
#         randomize_noise=False
#         )
#     # plot theta_pred

#     # with torch.no_grad():
#     #     observed_pred = theta_pred

#     #     train_x = opts.feats_dict.keys()
#     #     train_x = torch.tensor(list(train_x)).float()
#     #     train_y = torch.ones_like(train_x)

#     #     test_x = target_ages
#     #     # Initialize plot
#     #     f, ax = plt.subplots(1, 1, figsize=(4, 3))
#     #     # Get upper and lower confidence bounds
#     #     lower, upper = observed_pred.confidence_region()
#     #     # Plot training data as black stars
#     #     ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
#     #     # Plot predictive means as blue line
#     #     ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
#     #     # Shade between the lower and upper confidence bounds
#     #     ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
#     #     ax.set_ylim([-3, 3])
#     #     ax.legend(['Observed Data', 'Mean', 'Confidence'])

#     #     # save plot
#     #     plt.savefig(os.path.join('/playpen-nas-ssd/luchao/projects/SAM', 'theta_pred.png'))

#     return result_batch, latent_blended

# def run_on_batch_blender(inputs, net, net_global):
#     x = inputs

#     _, latent_local = net.forward(x, return_latents=True)
#     _, latent_global = net_global.forward(x, return_latents=True)
#     # returned latent styles: [b, 18, 512]
#     target_ages = x[:, -1, 0, 0]
#     # get latent styles for all ages
#     with torch.no_grad():
#         latent_global_ages = []
#         latent_local_ages = []
#         for age in range(0, 101, 10):
#             imgs = x[:, 0:3, :, :]
#             age = age / 100.
#             x_input = [torch.cat((img, age * torch.ones((1, img.shape[1], img.shape[2])).to('cuda')))
#                         for img in imgs]
#             x_input = torch.stack(x_input)
#             _, latent_global_age = net_global.forward(x_input, return_latents=True)
#             _, latent_local_age = net.forward(x_input, return_latents=True)
#             latent_global_ages.append(latent_global_age)
#             latent_local_ages.append(latent_local_age)
#         latent_global_ages = torch.stack(latent_global_ages) # [101, b, 18, 512]
#         latent_global_ages = latent_global_ages.permute(1, 0, 2, 3) # [b, 101, 18, 512]
#         latent_local_ages = torch.stack(latent_local_ages)
#         latent_local_ages = latent_local_ages.permute(1, 0, 2, 3)
#     latent_blended = net.blender(latent_local, latent_global, latent_local_ages, latent_global_ages, target_ages)

#     result_batch, _ = net.decoder(
#         [latent_blended], 
#         input_is_latent=True, 
#         randomize_noise=False
#         )
#     return result_batch, latent_blended

def helper(img_path):
    # print('Processing image: {} and saving to {}'.format(img_path, output_dir))
    # copy the image to single_img folder
    shutil.copy(img_path, output_dir_for_single_img)
    # * optional: align the image
    # try:
    #     aligned_image = run_alignment(img_path)
    # except Exception as e:
    #     print(f'Failed to align the image: {img_path}')
    #     print('Skip this image')
    #     return
    aligned_image = Image.open(img_path)
    img_transforms = EXPERIMENT_ARGS['transform']
    # print('Vision transforming done!')
    input_image = img_transforms(aligned_image)
    input_ages = img_path.split('/')[-1].split('.')[0].split('_')[0].split(' ')[0]
    # we'll run the image on multiple target ages 
    target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_transformers = [AgeTransformer(target_age=age) for age in target_ages]
    results = [np.array(aligned_image.resize((1024, 1024)))]

    results_age_pred = []
    results_id_similarity = []
    for i, age_transformer in enumerate(age_transformers):
        # print(f'Running on target age: {age_transformer.target_age}')
        with torch.no_grad():
            input_image_age = [age_transformer(input_image.cpu()).to('cuda')]
            input_image_age = torch.stack(input_image_age)

            result_tensor, result_style = run_on_batch(input_image_age, net)
            if blender:
                target_ages_blender = torch.tensor([target_ages[i] / 100]).float().to('cuda')
                input_ages_blender = torch.tensor([int(input_ages) / 100]).float().to('cuda')
                # age_diff = target_ages_blender - input_ages_blender
                result_tensor, result_style = run_on_batch_blender(input_image_age, net, net_global, target_ages=target_ages_blender)
                # result_tensor, result_style = run_on_batch_blender(input_image_age, net, net_global)
            # print(result_tensor.shape)
            result_tensor = result_tensor.squeeze(0) # (1, 3, 1024, 1024) -> (3, 1024, 1024)
            result_image = tensor2im(result_tensor)
            results.append(result_image)

            # also save result_style
            input_image_name = os.path.basename(img_path).split('.')[0]
            result_style = result_style.cpu().numpy()
            np.save(os.path.join(output_dir, f'latent_{input_image_name}_{age_transformer.target_age}.npy'), result_style)

            # aging evaluation
            target_age = target_ages[i]
            predict_age = aging_loss.extract_ages(result_tensor.unsqueeze(0)).item()
            results_age_pred.append(predict_age)

            # ID evaluation
            # todo: add training age range restriction: if taget_age< 30 or target_age > 70, skip
            # predict_img = img_transforms(result_image)
            # predict_img_id_feats = id_loss.extract_feats(predict_img.unsqueeze(0).to('cuda'))
            # if len([k for k in feats_dict.keys() if abs(k - target_age) <= 3]) == 0:
            #     print('No reference features found for target age: ', target_age)
            #     continue
            # reference_feats = torch.stack([feat[0] for k in feats_dict.keys() if abs(k - target_age) <= 3 for feat in feats_dict[k]])
            # # find the most similar reference feature to the predicted feature (cosine similarity)
            # id_similarity = nn.functional.cosine_similarity(predict_img_id_feats, reference_feats).max().item()
            # results_id_similarity.append(id_similarity)

    # ID evaluation
    input_img = copy.deepcopy(input_image)
    input_img_id_feats = id_loss.extract_feats(input_img.unsqueeze(0).to('cuda'))
    input_age = img_path.split('/')[-1].split('.')[0].split('_')[0].split(' ')[0]
    age_transformer = AgeTransformer(target_age=input_age)
    input_img_age = age_transformer(input_img.cpu()).to('cuda')
    result_tensor, _ = run_on_batch(input_img_age.unsqueeze(0), net)
    if blender:
        input_ages_blender = torch.tensor([int(input_age) / 100]).float().to('cuda')
        target_ages_blender = input_ages_blender
        # # result_tensor, _ = run_on_batch_blender(input_img_age.unsqueeze(0), net, net_global)
        result_tensor, _ = run_on_batch_blender(input_img_age.unsqueeze(0), net, net_global, target_ages=target_ages_blender)
        # result_tensor, _ = run_on_batch_blender(input_img_age.unsqueeze(0), net, net_global)
    result_tensor = result_tensor.squeeze(0)
    result_img = tensor2im(result_tensor)
    result_img_transformed = img_transforms(result_img)
    result_img_id_feats = id_loss.extract_feats(result_img_transformed.unsqueeze(0).to('cuda'))
    id_similarity_inversion = nn.functional.cosine_similarity(input_img_id_feats, result_img_id_feats).item()

    # save inversion results
    input_image_name = os.path.basename(img_path)
    input_img_ = np.array(aligned_image.resize((1024, 1024)))
    grid = make_grid([F.to_tensor(img) for img in [input_img_, result_img]], nrow=2)
    save_image(grid, os.path.join(output_dir, f'inversion_{input_image_name}'))
    # print('Saved to {}'.format(os.path.join(output_dir, f'inversion_{input_image_name}')))
    # save result_img into single_img folder
    save_image(F.to_tensor(result_img), os.path.join(output_dir_for_single_img, f'inversion_{input_image_name}'))


    # save aging results
    grid = make_grid([F.to_tensor(img) for img in results], nrow=12)
    # current_time = time.time()
    # # convert to yyyy-mm-dd-hh-mm-ss
    # current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(current_time))
    # save_image(grid, '/playpen-nas-ssd/luchao/projects/SAM/notebooks/images/{}.jpg'.format(current_time))
    input_image_name = os.path.basename(img_path)
    save_image(grid, os.path.join(output_dir, input_image_name))
    # print('Saved to {}'.format(os.path.join(output_dir, input_image_name)))
    # save each aging result into single_img folder
    for i, result in enumerate(results[1:]): # skip the original image
        save_image(F.to_tensor(result), os.path.join(output_dir_for_single_img, f'{input_image_name.split(".")[0]}_{target_ages[i]}.jpg'))


    return results_age_pred, id_similarity_inversion

def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--img_dir', type=str, help='path to the image directory')
    parser.add_argument('--desc', type=str,
                        default='', 
                        help='description of the inference')
    parser.add_argument('--model_path', type=str, help='path to the model (personalized?), default to pretrained SAM model')
    parser.add_argument('--blender', action='store_true', help='use blender')
    parser.add_argument('--output_dir', type=str, help='path to the output directory', default='/playpen-nas-ssd/luchao/projects/SAM/output_helper')
    # used for interpolation ID evaluation
    # parser.add_argument('--celeb_reference', type=str, help='celeb name', default='')
    return parser

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    img_dir = args.img_dir
    if args.desc:
        desc = args.desc
    else:
        desc_exp = args.model_path.split('/')[-4]
        desc_iter = args.model_path.split('/')[-1].split('.')[0]
        desc = f'{desc_exp}_{desc_iter}'
    # output_dir = '/playpen-nas-ssd/luchao/projects/SAM/output_helper'
    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, desc)
    os.makedirs(output_dir, exist_ok=True)
    output_dir_for_single_img = os.path.join(output_dir, 'single_img')
    os.makedirs(output_dir_for_single_img, exist_ok=True)

    def iterate_over_img_dir(img_dir):
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                if any(file.endswith(extension) for extension in IMG_EXTENSIONS):
                    yield os.path.join(root, file)

    img_paths = sorted(list(iterate_over_img_dir(img_dir)))


    # import multiprocessing
    # print('Start processing {} images'.format(len(img_paths)))
    # with multiprocessing.Pool(4) as p:
    #     p.map(helper, [img_path for img_path in img_paths])
    # # helper(img_path)
    # print('Done!')

    EXPERIMENT_TYPE = 'ffhq_aging'
    # ! no need to be aligned - can be in-the-wild
    # img_path = '/playpen-nas-ssd/luchao/data/age/robert/11.jpeg'
    # ! todo: change the model path
    EXPERIMENT_DATA_ARGS = {
        "ffhq_aging": {
            "model_path": "/playpen-nas-ssd/luchao/projects/SAM/pretrained_models/sam_ffhq_aging.pt",
            # ! todo: change the model path here
            # "model_path": '/playpen-nas-ssd/luchao/projects/SAM/training-runs/robert/00000/checkpoints/iteration_10000.pt',
            # "image_path": img_path,
            "transform": transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
    }

    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[EXPERIMENT_TYPE]
    model_path = EXPERIMENT_ARGS['model_path'] if args.model_path is None else args.model_path
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    # ! important: change checkpoint_path to model_path to load the model
    opts['checkpoint_path'] = model_path

    if opts.get('exp_dir', None) is not None:
        # copy opt.json to output_dir
        opt_file = os.path.join(opts['exp_dir'], 'opt.json')
        if os.path.exists(opt_file):
            os.system('cp {} {}'.format(opt_file, output_dir))
        else:
            print("WARNING: opt.json not found in checkpoint's exp_dir")

    pprint.pprint(opts)
    opts = Namespace(**opts)
    net = pSp(opts)

    # ------------------------------- for eval only ------------------------------ #
    aging_loss = AgingLoss(opts)
    id_loss = id_loss.IDLoss().to('cuda').eval()
    # ----------------------------- for blender only ----------------------------- #
    # blender network weights are in ckpt['blender]
    # reload encoder to get global weights
    @staticmethod
    def __get_keys(d, name):
        if 'state_dict' in d:
            d = d['state_dict']
        d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
        return d_filt
    def load_global_encoder(net):
        model_path_global = EXPERIMENT_ARGS['model_path']
        print(f'Loading global encoder from {model_path_global}')
        ckpt_global = torch.load(model_path_global, map_location='cpu')
        net.encoder.load_state_dict(__get_keys(ckpt_global, 'encoder'), strict=False)
        return net
    def load_global_decoder(net):
        model_path_global = EXPERIMENT_ARGS['model_path']
        print(f'Loading global decoder from {model_path_global}')
        ckpt_global = torch.load(model_path_global, map_location='cpu')
        net.decoder.load_state_dict(__get_keys(ckpt_global, 'decoder'), strict=True)
        return net
    blender = args.blender
    if blender:
        # make a copy of the original net
        net_global = copy.deepcopy(net)
        net_global = load_global_encoder(net_global)
        net_global = load_global_decoder(net_global)
        net_global.eval()
        net_global.cuda()
        print('Global model successfully loaded using blender!')

    net.eval()
    net.cuda()


    age_results = {}
    id_results = {}
    for img_path in tqdm(img_paths):
        helper_output = helper(img_path)
        if helper_output is None:
            continue
        else:
            results_age_pred, inversion_id_similarity = helper_output
        # results
        age_results[img_path] = results_age_pred
        id_results[img_path] = inversion_id_similarity

    # average over results across images
    age_results_avg = []
    id_results_avg = []
    for img_path, results in age_results.items():
        age_results_avg.append(results)
    for img_path, results in id_results.items():
        id_results_avg.append(results)
    age_results_avg = np.mean(age_results_avg, axis=0)
    target_ages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    age_error_avg = abs(age_results_avg - target_ages)
    # format to 2-decimals
    age_error_avg = np.round(age_error_avg, 2)
    id_results_avg = np.mean(id_results_avg)
    # print('Average age error (abs) for each target year: ', age_error_avg)
    # print(f'Average age error (abs): \t', np.mean(age_error_avg))
    # print(f'Average inverted ID similarity: \t', id_results_avg)
    # print('Done!')

    # save results into a txt file
    with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
        f.write('Average age error (abs) for each target year: \n')
        f.write(str(age_error_avg) + '\n')
        f.write(f'Average age error (abs): \n')
        f.write(str(np.mean(age_error_avg)) + '\n')
        f.write(f'Average inverted ID similarity: \n')
        f.write(str(id_results_avg) + '\n')