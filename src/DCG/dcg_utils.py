import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as func


def _retrieve_crop(x_original_pytorch, crop_positions, crop_method, parameters):
        """
        Function that takes in the original image and cropping position and returns the crops
        :param x_original_pytorch: PyTorch Tensor array (N,C,H,W)
        :param crop_positions:
        :return:
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = parameters["crop_shape"]

        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        if parameters["device_type"] == "gpu":
            device = torch.device("cuda:{}".format(parameters["gpu_number"]))
            output = output.to(device)
        for i in range(batch_size):
            for j in range(num_crops):
                crop_pytorch(x_original_pytorch[i, 0, :, :],
                                                    parameters["crop_shape"],
                                                    crop_positions[i,j,:],
                                                    output[i,j,:,:],
                                                    method=crop_method)
        return output

def _convert_crop_position(crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original
        :param crops_x_small: N, k*c, 2 numpy matrix
        :param cam_size: (h,w)
        :param x_original: N, C, H, W pytorch variable
        :return: N, k*c, 2 numpy matrix
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np.max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np.min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np.max(top_k_prop_y) <= 1.0, "top_k_prop_y >= 1.0"
        assert np.min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = np.expand_dims(np.around(top_k_prop_x * H), -1)
        top_k_interpolate_y = np.expand_dims(np.around(top_k_prop_y * W), -1)
        top_k_interpolate_2d = np.concatenate([top_k_interpolate_x, top_k_interpolate_y], axis=-1)
        return top_k_interpolate_2d
    
def get_max_window(input_image, window_shape, pooling_logic="avg"):
    """
    Function that makes a sliding window of size window_shape over the
    input_image and return the UPPER_LEFT corner index with max sum
    :param input_image: N*C*H*W
    :param window_shape: h*w
    :return: N*C*2 tensor
    """
    N, C, H, W = input_image.size()
    if pooling_logic == "avg":
        # use average pooling to locate the window sums
        pool_map = torch.nn.functional.avg_pool2d(input_image, window_shape, stride=1)
    elif pooling_logic in ["std", "avg_entropy"]:
        # create sliding windows
        output_size = (H - window_shape[0] + 1, W - window_shape[1] + 1)
        sliding_windows = F.unfold(input_image, kernel_size=window_shape).view(N,C, window_shape[0]*window_shape[1], -1)
        # apply aggregation function on each sliding windows
        if pooling_logic == "std":
            agg_res = sliding_windows.std(dim=2, keepdim=False)
        elif pooling_logic == "avg_entropy":
            agg_res = -sliding_windows*torch.log(sliding_windows)-(1-sliding_windows)*torch.log(1-sliding_windows)
            agg_res = agg_res.mean(dim=2, keepdim=False)
        # merge back
        pool_map = F.fold(agg_res, kernel_size=(1, 1), output_size=output_size)
    _, _, _, W_map = pool_map.size()
    # transform to linear and get the index of the max val locations
    _, max_linear_idx = torch.max(pool_map.view(N, C, -1), -1)
    # convert back to 2d index
    #max_idx_x = max_linear_idx // W_map
    max_idx_x = torch.div(max_linear_idx, W_map, rounding_mode='trunc')
    max_idx_y = max_linear_idx - max_idx_x * W_map
    # put together the 2d index
    upper_left_points = torch.cat([max_idx_x.unsqueeze(-1), max_idx_y.unsqueeze(-1)], dim=-1)
    return upper_left_points

def generate_mask_uplft(input_image, window_shape, upper_left_points, gpu_number):
    """
    Function that generates mask that sets crops given upper_left
    corners to 0
    :param input_image:
    :param window_shape:
    :param upper_left_points:
    """
    N, C, H, W = input_image.size()
    window_h, window_w = window_shape
    # get the positions of masks
    mask_x_min = upper_left_points[:,:,0]
    mask_x_max = upper_left_points[:,:,0] + window_h
    mask_y_min = upper_left_points[:,:,1]
    mask_y_max = upper_left_points[:,:,1] + window_w
    # generate masks
    mask_x = Variable(torch.arange(0, H).view(-1, 1).repeat(N, C, 1, W))
    mask_y = Variable(torch.arange(0, W).view(1, -1).repeat(N, C, H, 1))
    if gpu_number is not None:
        device = torch.device("cuda:{}".format(gpu_number))
        mask_x = mask_x.to(device)
        mask_y = mask_y.to(device)
    x_gt_min = mask_x.float() >= mask_x_min.unsqueeze(-1).unsqueeze(-1).float()
    x_ls_max = mask_x.float() < mask_x_max.unsqueeze(-1).unsqueeze(-1).float()
    y_gt_min = mask_y.float() >= mask_y_min.unsqueeze(-1).unsqueeze(-1).float()
    y_ls_max = mask_y.float() < mask_y_max.unsqueeze(-1).unsqueeze(-1).float()

    # since logic operation is not supported for variable
    # I used * for logic ANd
    selected_x = x_gt_min * x_ls_max
    selected_y = y_gt_min * y_ls_max
    selected = selected_x * selected_y
    mask = 1 - selected.float()
    return mask

