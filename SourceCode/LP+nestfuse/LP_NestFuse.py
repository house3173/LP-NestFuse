import os.path as path
import skimage.io as io
import numpy as np
import scipy as sp
from skimage import color
from skimage import util
import matplotlib.pyplot as plt
import os
from skimage import img_as_ubyte
import torch
from torch.autograd import Variable
from args_fusion import args
import LP as lp
import NestFuse_model as modelNestFuse
import imageio
import glob
import matplotlib.gridspec as gridspec
import time 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_fused_pyramid(fused_pyramid, output_folder, identifier):
    for i, layer in enumerate(fused_pyramid):
        norm_layer = (layer - np.min(layer)) / (np.max(layer) - np.min(layer))  # Normalize to 0-1 range
        norm_layer = (norm_layer * 255).astype(np.uint8)  # Convert to uint8
        output_path = os.path.join(output_folder, f"fused_layer_{i+1}.png")
        imageio.imwrite(output_path, norm_layer)
        
def plot_fused_pyramid(pyramid, title_prefix):
    num_levels = len(pyramid)
    plt.figure(figsize=(10, 10))
    for i, img in enumerate(pyramid):
        plt.subplot(1, num_levels, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f'{title_prefix} Level {i+1}')
        plt.axis('off')
    plt.show()

def normalize_all_layers(pyramid, target_min=0, target_max=1):
    normalized_pyramid = []
    for layer in pyramid:
        normalized_layer = normalize_to_dynamic_range(layer, target_min, target_max)
        normalized_pyramid.append(normalized_layer)
    return normalized_pyramid

def normalize_to_dynamic_range(layer, target_min=0, target_max=1):
    layer_min = layer.min()
    layer_max = layer.max()
    if layer_max == layer_min:
        return np.full(layer.shape, (target_min + target_max) / 2)
    else:
        normalized_layer = (layer - layer_min) / (layer_max - layer_min)
        normalized_layer = normalized_layer * (target_max - target_min) + target_min
        return normalized_layer

def get_test_image(image):
    images = []
    image = normalize_to_dynamic_range(image, 0, 255)
    image = np.reshape(image, [1, image.shape[0], image.shape[1]])
    images.append(image)
    images = np.stack(images, axis=0)
    image = torch.from_numpy(images).float().to(device)
    return image

def process_image_pair(ir_path, vis_path, model, output_path, fs_type, pair_id, prefix):
    img_ir = io.imread(ir_path)
    img_vis = io.imread(vis_path)

    if len(img_ir.shape) == 3:
        img_ir = color.rgb2gray(img_ir)
    if len(img_vis.shape) == 3:
        img_vis = color.rgb2gray(img_vis)
    
    img_ir_origin = util.img_as_float(img_ir)
    img_vis_origin = util.img_as_float(img_vis)
    max_levels = 5

    pyramid_ir = lp.laplacian_pyramid.decompose(img_ir_origin, levels=max_levels)
    pyramid_vis = lp.laplacian_pyramid.decompose(img_vis_origin, levels=max_levels)
    # output_folder = f'./laplacian_pyramid_{prefix}'
    # output_folder_ir = path.join(output_folder, f'{prefix}_ir/IR{pair_id}').replace(".png", "")
    # output_folder_vis = path.join(output_folder, f'{prefix}_vis/VIS{pair_id}').replace(".png", "")
    # os.makedirs(output_folder_ir, exist_ok=True)
    # os.makedirs(output_folder_vis, exist_ok=True)
    # lp.save_pyramid(pyramid_ir, output_folder_ir, "IR")
    # lp.save_pyramid(pyramid_vis, output_folder_vis, "VIS")
    
    # Sử dụng mô hình NestFuse để hợp nhất thành phần chi tiết
    with torch.no_grad():
        pyramid_detail_fusion = [None] * (max_levels - 1)
        for i in range(1, max_levels):
            pyramid_detail_ir = get_test_image(pyramid_ir[i])
            pyramid_detail_vis = get_test_image(pyramid_vis[i])
            img_ir = Variable(pyramid_detail_ir, requires_grad=False)
            img_vi = Variable(pyramid_detail_vis, requires_grad=False)

            en_r = model.encoder(img_ir)
            en_v = model.encoder(img_vi)
            f = model.fusion(en_r, en_v, fs_type)
            img_fusion_list = model.decoder_eval(f)

            for img_fusion in img_fusion_list:
                pyramid_detail_fusion[i - 1] = img_fusion.squeeze(0).squeeze(0).cpu().numpy()
                pyramid_detail_fusion[i - 1] = normalize_to_dynamic_range(pyramid_detail_fusion[i - 1], 0, 1)

    fused_pyramid = [None] * max_levels
    fused_pyramid[0] = lp.pyramid_fusion_base_level(pyramid_ir[0], pyramid_vis[0])
    fused_pyramid[1:] = pyramid_detail_fusion
    
    # output_subfolder = os.path.join(f"./pyramid_fused_{prefix}", f"{prefix}_pair_{pair_id}").replace(".png", "")
    # if not os.path.exists(output_subfolder):
    #     os.makedirs(output_subfolder)
    # save_fused_pyramid(fused_pyramid, output_subfolder, f"pair_{pair_id}")
    
    img_fused = lp.laplacian_pyramid.reconstruct(fused_pyramid)
    io.imsave(output_path, img_as_ubyte(np.clip(img_fused, 0, 1)))
    print(output_path)
    
    # Plotting images
    # plot_images(img_ir_origin, img_vis_origin, pyramid_ir, pyramid_vis, fused_pyramid, img_fused)


def plot_images(img_ir, img_vis, pyramid_ir, pyramid_vis, fused_pyramid, img_fused):
    num_levels = len(pyramid_ir)
    fig = plt.figure(figsize=(100, 100))  # Tăng kích thước hình

    # Create grid layout
    gs = gridspec.GridSpec(6, num_levels + 1, width_ratios=[0.2] + [1]*num_levels, wspace=0.0, hspace=0.3)

    # Titles for each row
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.text(0.5, 0.5, 'Origin', ha='center', va='center')

    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    ax.text(0.5, 0.5, 'IR Decompose', ha='center', va='center')

    ax = fig.add_subplot(gs[2, 0])
    ax.axis('off')
    ax.text(0.5, 0.5, 'VIS Decompose', ha='center', va='center')

    ax = fig.add_subplot(gs[3, 0])
    ax.axis('off')
    ax.text(0.5, 0.5, 'Fused Decompose', ha='center', va='center')

    ax = fig.add_subplot(gs[4, 0])
    ax.axis('off')
    ax.text(0.5, 0.5, 'Output', ha='center', va='center')

    # Images for each level
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(img_ir, cmap='gray')
    ax.set_title('IR Image')
    ax.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(img_vis, cmap='gray')
    ax.set_title('VIS Image')
    ax.axis('off')

    for i in range(num_levels):
        ax = fig.add_subplot(gs[1, i + 1])
        ax.imshow(pyramid_ir[i], cmap='gray')
        ax.set_title(f'IR Level {i+1}')
        ax.axis('off')

    for i in range(num_levels):
        ax = fig.add_subplot(gs[2, i + 1])
        ax.imshow(pyramid_vis[i], cmap='gray')
        ax.set_title(f'VIS Level {i+1}')
        ax.axis('off')

    for i in range(num_levels):
        ax = fig.add_subplot(gs[3, i + 1])
        ax.imshow(fused_pyramid[i], cmap='gray')
        ax.set_title(f'Fused Level {i+1}')
        ax.axis('off')

    ax = fig.add_subplot(gs[4, 1])
    ax.imshow(img_fused, cmap='gray')
    ax.set_title('Fused Image')
    ax.axis('off')

    plt.show()


def find_matching_images(ir_folder, vis_folder):
    ir_images = {os.path.basename(path): path for path in glob.glob(os.path.join(ir_folder, '*.png'))}
    vis_images = {os.path.basename(path): path for path in glob.glob(os.path.join(vis_folder, '*.png'))}
    common_names = sorted(set(ir_images.keys()).intersection(set(vis_images.keys())))
    matching_pairs = [(ir_images[name], vis_images[name]) for name in common_names]
    return matching_pairs

def process_images(ir_folder, vis_folder, output_folder, model, fs_type, prefix):
    os.makedirs(output_folder, exist_ok=True)
    image_pairs = find_matching_images(ir_folder, vis_folder)
    for ir_path, vis_path in image_pairs:
        output_filename = os.path.basename(ir_path)
        output_path = os.path.join(output_folder, output_filename)
        process_image_pair(ir_path, vis_path, model, output_path, fs_type, output_filename, prefix)
        print(f"Processed and saved fusion image to {output_path}")

def main():
    dataset_choice = 'TEST_3'

    if dataset_choice == 'MSRS':
        ir_folder = './images/MSRS/MSRS/ir'
        vis_folder = './images/MSRS/MSRS/vi'
        output_folder = './MSRS_results'
        prefix = 'msrs'
    elif dataset_choice == 'TNO':
        ir_folder = './images/TNO/ir'
        vis_folder = './images/TNO/vi'
        output_folder = './TNO_results'
        prefix = 'tno'
    elif dataset_choice == 'TEST':
        ir_folder = 'C:/PVH/SourceCode/LP+nestfuse/images/TEST/ir'
        vis_folder = 'C:/PVH/SourceCode/LP+nestfuse/images/TEST/vi'
        output_folder = 'C:/PVH/SourceCode/LP+nestfuse/TEST_results'
        prefix = 'test'
    elif dataset_choice == 'TEST_2':
        ir_folder = 'C:/PVH/SourceCode/LP+nestfuse/images/TEST_2/ir'
        vis_folder = 'C:/PVH/SourceCode/LP+nestfuse/images/TEST_2/vi'
        output_folder = 'C:/PVH/SourceCode/LP+nestfuse/TEST_2_results'
        prefix = 'test_2'
    elif dataset_choice == 'TEST_3':
        ir_folder = 'C:/PVH/SourceCode/LP+nestfuse/TNO/ir'
        vis_folder = 'C:/PVH/SourceCode/LP+nestfuse/TNO/vi'
        output_folder = 'C:/PVH/SourceCode/LP+nestfuse/TNO_results/PVH'
        prefix = 'TNO_TEST'

    os.makedirs(output_folder, exist_ok=True)
    fs_type = 'attention_avg'
    print("Starting load model")
    model_path = args.model_default
    model = modelNestFuse.load_model(model_path, False)
    model.to(device)
    start_time = time.time()
    process_images(ir_folder, vis_folder, output_folder, model, fs_type, prefix)
    end_time = time.time()
    print(f"Total time: {(end_time - start_time):.6f} seconds")
    print("Finished processing images")

if __name__ == "__main__":
    main()
