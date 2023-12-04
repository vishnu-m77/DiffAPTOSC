import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import src.DCG.utils as utils
import torchvision
import torchvision.models
from torchvision.models.resnet import conv3x3, resnet18, resnet50

# None of the classes are being used in the implementation of DCG.

class AbstractMILUnit:
    """
    An abstract class that represents an MIL unit module
    """
    def __init__(self, params, parent_module):
        self.params = params
        self.parent_module = parent_module


# class PostProcessingStandard(nn.Module):
class Saliency_Map(nn.Module):
    """
    Unit in Global Network that takes in x_out and produce saliency maps
    """
    def __init__(self, params):
        super(Saliency_Map, self).__init__()
        # map all filters to output classes
        self.gn_conv_last = nn.Conv2d(params["post_processing_dim"]*4,
                                      params["num_classes"],
                                      (1, 1), bias=False)

    def forward(self, x_out):
        out = self.gn_conv_last(x_out)
        return out
    
class DownsampleNetwork(nn.Module):
    """
    Downsampling using ResNet V1
    First conv is 7*7, stride 2, padding 3, cut 1/2 resolution
    """
    def __init__(self):
        super(DownsampleNetwork, self).__init__()
        self.f = []
        # backbone = resnet50(pretrained=True)
        backbone = resnet50(weights='DEFAULT')
        # weights=ResNet18_Weights.DEFAULT
        # backbone = resnet18(pretrained=True)
        
        for name, module in backbone.named_children():
            if name != 'fc' and name != 'avgpool':
                self.f.append(module)
        # print(self.f)
        # encoder
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        last_feature_map = self.f(x)
        # print(last_feature_map.shape)
        return last_feature_map


class GlobalNetwork(AbstractMILUnit):
    """
    Implementation of Global Network using ResNet-22
    """
    def __init__(self, params, parent_module):
        super(GlobalNetwork, self).__init__(params, parent_module)
        # downsampling-branch
        # if "use_v1_global" in params and params["use_v1_global"]:
        self.downsampling_branch = DownsampleNetwork()
        # post-processing
        self.saliency_map = Saliency_Map(params)

    def add_layers(self):
        self.parent_module.ds_net = self.downsampling_branch
        self.parent_module.left_postprocess_net = self.saliency_map

    def forward(self, x):
        # retrieve results from downsampling network at all 4 levels
        last_feature_map = self.downsampling_branch.forward(x)
        # feed into postprocessing network
        cam = self.saliency_map.forward(last_feature_map)
        return last_feature_map, cam

def global_res(x, params):
    downsampling_branch = DownsampleNetwork()
    # post-processing
    saliency_map = Saliency_Map(params)
    # retrieve results from downsampling network at all 4 levels
    last_feature_map = downsampling_branch.forward(x)
    # feed into postprocessing network
    cam = saliency_map.forward(last_feature_map)
    return last_feature_map, cam


class TopTPercentAggregationFunction(AbstractMILUnit):
    """
    An aggregator that uses the SM to compute the y_global.
    Use the sum of topK value
    """
    def __init__(self, params, parent_module):
        super(TopTPercentAggregationFunction, self).__init__(params, parent_module)
        self.percent_t = params["percent_t"]
        self.parent_module = parent_module

    def forward(self, cam):
        batch_size, num_class, H, W = cam.size()
        cam_flatten = cam.view(batch_size, num_class, -1)
        top_t = int(round(W*H*self.percent_t))
        selected_area = cam_flatten.topk(top_t, dim=2)[0]
        return selected_area.mean(dim=2)

def aggregator(percent_t, cam):
    batch_size, num_class, H, W = cam.size()
    cam_flatten = cam.view(batch_size, num_class, -1)
    top_t = int(round(W*H*percent_t))
    selected_area = cam_flatten.topk(top_t, dim=2)[0]
    return selected_area.mean(dim=2)


class RetrieveROIModule(AbstractMILUnit):
    """
    A Regional Proposal Network instance that computes the locations of the crops
    Greedy select crops with largest sums
    """
    def __init__(self, params, parent_module):
        super(RetrieveROIModule, self).__init__(params, parent_module)
        self.crop_method = "upper_left"
        self.num_crops_per_class = params["K"]
        self.crop_shape = params["crop_shape"]
        self.gpu_number = None if params["device"]!="gpu" else params["gpu_number"]

    def forward(self, x_original, cam_size, h_small):
        """
        Function that use the low-res image to determine the position of the high-res crops
        :param x_original: N, C, H, W pytorch tensor
        :param cam_size: (h, w)
        :param h_small: N, C, h_h, w_h pytorch tensor
        :return: N, num_classes*k, 2 numpy matrix; returned coordinates are corresponding to x_small
        """
        # retrieve params
        _, _, H, W = x_original.size()
        (h, w) = cam_size
        N, C, h_h, w_h = h_small.size()
        #print(h_small.size())
        # make sure that the size of h_small == size of cam_size
        # assert h_h == h, "h_h!=h"
        # assert w_h == w, "w_h!=w"

        # adjust crop_shape since crop shape is based on the original image
        crop_x_adjusted = int(np.round(self.crop_shape[0] * h / H))
        crop_y_adjusted = int(np.round(self.crop_shape[1] * w / W))
        crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

        # greedily find the box with max sum of weights
        current_images = h_small
        all_max_position = []
        # combine channels
        max_vals = current_images.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
        min_vals = current_images.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
        range_vals = max_vals - min_vals
        normalize_images = current_images - min_vals
        normalize_images = normalize_images / range_vals
        current_images = normalize_images.sum(dim=1, keepdim=True)

        for _ in range(self.num_crops_per_class):
            max_pos = utils.get_max_window(current_images, crop_shape_adjusted, "avg")
            all_max_position.append(max_pos)
            mask = utils.generate_mask_uplft(current_images, crop_shape_adjusted, max_pos, self.gpu_number)
            current_images = current_images * mask
        return torch.cat(all_max_position, dim=1).data.cpu().numpy()


def retrieve_roi(x_original, cam_size, h_small, params):
    """
    Function that use the low-res image to determine the position of the high-res crops
    :param x_original: N, C, H, W pytorch tensor
    :param cam_size: (h, w)
    :param h_small: N, C, h_h, w_h pytorch tensor
    :return: N, num_classes*k, 2 numpy matrix; returned coordinates are corresponding to x_small
    """
    gpu_number = None if params["device"]!="gpu" else params["gpu_number"]
    # retrieve params
    _, _, H, W = x_original.size()
    (h, w) = cam_size
    N, C, h_h, w_h = h_small.size()
    #print(h_small.size())
    # make sure that the size of h_small == size of cam_size
    assert h_h == h, "h_h!=h"
    assert w_h == w, "w_h!=w"
    # adjust crop_shape since crop shape is based on the original image
    crop_x_adjusted = int(np.round(params["crop_shape"][0] * h / H))
    crop_y_adjusted = int(np.round(params["crop_shape"][1] * w / W))
    crop_shape_adjusted = (crop_x_adjusted, crop_y_adjusted)

    # greedily find the box with max sum of weights
    all_max_position = []
    # combine channels
    max_vals = h_small.view(N, C, -1).max(dim=2, keepdim=True)[0].unsqueeze(-1)
    min_vals = h_small.view(N, C, -1).min(dim=2, keepdim=True)[0].unsqueeze(-1)
    range_vals = max_vals - min_vals
    normalize_images = h_small - min_vals
    normalize_images = normalize_images / range_vals
    h_small = normalize_images.sum(dim=1, keepdim=True)

    for _ in range(params["K"]):
        max_pos = utils.get_max_window(h_small, crop_shape_adjusted, "avg")
        all_max_position.append(max_pos)
        mask = utils.generate_mask_uplft(h_small, crop_shape_adjusted, max_pos, gpu_number)
        h_small = h_small * mask
    return torch.cat(all_max_position, dim=1).data.cpu().numpy()
    
class LocalNetwork(AbstractMILUnit):
    """
    The local network that takes a crop and computes its hidden representation
    Use ResNet
    """
    def add_layers(self):
        """
        Function that add layers to the parent module that implements nn.Module
        :return:
        """
        self.parent_module.dn_resnet = DownsampleNetwork()

    def forward(self, x_crop):
        """
        Function that takes in a single crop and return the hidden representation
        :param x_crop: (N,C,h,w)
        :return:
        """
        # forward propagte using ResNet
        res = self.parent_module.dn_resnet(x_crop.expand(-1, 3, -1 , -1))
        # global average pooling
        res = res.mean(dim=2).mean(dim=2)
        return res

def local_res(x_crop):
    """
    The local network that takes a crop and computes its hidden representation
    Use ResNet
    Function that takes in a single crop and return the hidden representation
    :param x_crop: (N,C,h,w)
    :return:
    """
    net = DownsampleNetwork()
    # forward propagte using ResNet
    res = net(x_crop.expand(-1, 3, -1 , -1))
    # global average pooling
    res = res.mean(dim=2).mean(dim=2)
    return res


class AttentionModule(AbstractMILUnit):
    """
    The attention module takes multiple hidden representations and compute the attention-weighted average
    Use Gated Attention Mechanism in https://arxiv.org/pdf/1802.04712.pdf
    """
    def add_layers(self):
        """
        Function that add layers to the parent module that implements nn.Module
        :return:
        """
        # The gated attention mechanism
        self.parent_module.mil_attn_V = nn.Linear(512*4, 128, bias=False)
        self.parent_module.mil_attn_U = nn.Linear(512*4, 128, bias=False)
        self.parent_module.mil_attn_w = nn.Linear(128, 1, bias=False)
        # classifier
        self.parent_module.classifier = nn.Linear(512*4, self.params["num_classes"], bias=False)

    def forward(self, h_crops):
        """
        Function that takes in the hidden representations of crops and use attention to generate a single hidden vector
        :param h_small:
        :param h_crops:
        :return:
        """
        batch_size, num_crops, h_dim = h_crops.size()
        h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
        # calculate the attn score
        attn_projection = torch.sigmoid(self.parent_module.mil_attn_U(h_crops_reshape)) * \
                          torch.tanh(self.parent_module.mil_attn_V(h_crops_reshape))
        attn_score = self.parent_module.mil_attn_w(attn_projection)
        # use softmax to map score to attention
        attn_score_reshape = attn_score.view(batch_size, num_crops)
        attn = func.softmax(attn_score_reshape, dim=1)

        # final hidden vector
        z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)

        # map to the final layer
        y_crops = self.parent_module.classifier(z_weighted_avg)
        return z_weighted_avg, attn, y_crops
    
    
def attention(h_crops, params):
    """
    Function that add layers to the parent module that implements nn.Module
    :return:
    """
    # The gated attention mechanism
    mil_attn_V = nn.Linear(512*4, 128, bias=False)
    mil_attn_U = nn.Linear(512*4, 128, bias=False)
    mil_attn_w = nn.Linear(128, 1, bias=False)
    # classifier
    classifier = nn.Linear(512*4, params["num_classes"], bias=False)

    """
    Function that takes in the hidden representations of crops and use attention to generate a single hidden vector
    :param h_small:
    :param h_crops:
    :return:
    """
    batch_size, num_crops, h_dim = h_crops.size()
    h_crops_reshape = h_crops.view(batch_size * num_crops, h_dim)
    # calculate the attn score
    attn_projection = torch.sigmoid(mil_attn_U(h_crops_reshape)) * \
                        torch.tanh(mil_attn_V(h_crops_reshape))
    attn_score = mil_attn_w(attn_projection)
    # use softmax to map score to attention
    attn_score_reshape = attn_score.view(batch_size, num_crops)
    attn = func.softmax(attn_score_reshape, dim=1)

    # final hidden vector
    z_weighted_avg = torch.sum(attn.unsqueeze(-1) * h_crops, 1)

    # map to the final layer
    y_crops = classifier(z_weighted_avg)
    return y_crops
    

