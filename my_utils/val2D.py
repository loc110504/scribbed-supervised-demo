import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from matplotlib import pyplot as plt
def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    else:
        return 0, 0
    

def test_single_volume_test(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output = net(input)
                if len(output[0].shape) == 4:
                    output = output[0]
                out = torch.argmax(torch.softmax(
                    output, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list

def test_single_volume(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                out = torch.argmax(torch.softmax(
                    net(input), dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(
                net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_ds(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main, _, _, _ = net(input)
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list


def test_single_volume_cct_tree(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction_1 = np.zeros_like(label)
        prediction_2 = np.zeros_like(label)
        prediction_mix = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_1,output_2,output_mix = net(input)
                # output_main = net(input)[0]
                out_1 = torch.argmax(torch.softmax(
                    output_1, dim=1), dim=1).squeeze(0)
                out_1 = out_1.cpu().detach().numpy()
                out_2 = torch.argmax(torch.softmax(
                    output_2, dim=1), dim=1).squeeze(0)
                out_2 = out_2.cpu().detach().numpy()
                out_mix = torch.argmax(torch.softmax(
                    output_mix, dim=1), dim=1).squeeze(0)
                out_mix = out_mix.cpu().detach().numpy()
                pred_1 = zoom(
                    out_1, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction_1[ind] = pred_1

                pred_2 = zoom(
                    out_2, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction_2[ind] = pred_2

                pred_mix = zoom(
                    out_mix, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction_mix[ind] = pred_mix

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list_1, metric_list_2, metric_list_mix = [],[],[]
    for i in range(1, classes):
        metric_list_1.append(calculate_metric_percase(
            prediction_1 == i, label == i))
    for i in range(1, classes):
        metric_list_2.append(calculate_metric_percase(
            prediction_2 == i, label == i))
    for i in range(1, classes):
        metric_list_mix.append(calculate_metric_percase(
            prediction_mix == i, label == i))
    return metric_list_1, metric_list_2, metric_list_mix

def test_single_volume_cct(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.squeeze(0).cpu().detach(
    ).numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                # print(np.unique(output_main.cpu().detach().numpy()))
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                # plt.figure(figsize=(10,10))
                # plt.imshow(input[0][0].cpu().detach().numpy(),cmap='gray')
                # plt.imshow((out==1).cpu().detach().numpy(),alpha= 0.5)
                # plt.axis("off")
                # plt.savefig(f'data/111/{111}_mask_LV.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()

                # plt.figure(figsize=(10,10))
                # plt.imshow(input[0][0].cpu().detach().numpy(),cmap='gray')
                # plt.imshow((out==2).cpu().detach().numpy(),alpha= 0.5)
                # plt.axis("off")
                # plt.savefig(f'data/111/{111}_mask_MYO.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()

                # plt.figure(figsize=(10,10))
                # plt.imshow(input[0][0].cpu().detach().numpy(),cmap='gray')
                # plt.imshow((out==3).cpu().detach().numpy(),alpha= 0.5)
                # plt.axis("off")
                # plt.savefig(f'data/111/{111}_mask_RV.png', bbox_inches='tight', pad_inches=0) 
                # plt.close()
                # print(np.unique(out.cpu().detach().numpy()))
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list




def test_single_volume_cct_cell(image, label, net, classes, patch_size=[256, 256]):
    image, label = image.cpu().detach(
    ).numpy(), label.cpu().detach().numpy()
    if len(image.shape) == 4:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind]
            _,x, y = slice.shape[0], slice.shape[1]
            slice = zoom(
                slice, (1,patch_size[0] / x, patch_size[1] / y), order=0)
            input = torch.from_numpy(slice).unsqueeze(
                0).float().cuda()
            net.eval()
            with torch.no_grad():
                output_main = net(input)[0]
                out = torch.argmax(torch.softmax(
                    output_main, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                pred = zoom(
                    out, (x / patch_size[0], y / patch_size[1]), order=0)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            output_main, _, _, _ = net(input)
            out = torch.argmax(torch.softmax(
                output_main, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(
            prediction == i, label == i))
    return metric_list