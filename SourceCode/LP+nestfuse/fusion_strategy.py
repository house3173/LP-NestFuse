import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils


EPSILON = 1e-5


# attention fusion strategy, average based on weight maps
def attention_fusion_weight_old(tensor1, tensor2, p_type):
    # avg, max, nuclear
    f_channel = channel_fusion(tensor1, tensor2,  p_type)
    f_spatial = spatial_fusion(tensor1, tensor2)

    tensor_f = (f_channel + f_spatial) / 2
    return tensor_f


# select channel
def channel_fusion(tensor1, tensor2, p_type):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# channel attention
def channel_attention(tensor, pooling_type='avg'):
    # global pooling
    shape = tensor.size()
    pooling_function = F.avg_pool2d

    if pooling_type == 'attention_avg':
        pooling_function = F.avg_pool2d
    elif pooling_type == 'attention_max':
        pooling_function = F.max_pool2d
    elif pooling_type == 'attention_nuclear':
        pooling_function = nuclear_pooling
    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial


# pooling function
def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors


'''
    Build the model for fusion with Parallel-Channel Attention and Cross-Spatial Attention
    Step 1: Parallel-Channel Attention apply for each tensor (def channel_attention -> softmax -> tensor_1 * weight_1; tensor_2 * weight_2)
    Step 2: Cross-Spatial Attention apply for each tensor (def spatial_attention -> softmax -> tensor_1 * weight_2 + tensor_2 * weight_1)
'''
def attention_fusion_weight_1(tensor1, tensor2, p_type):
    shape = tensor1.size()

    # Step 1: Parallel-Channel Attention
    # 1.1. Calculate the global pooling for each tensor
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # 1.2. Calculate the weight map for each tensor (softmax function) 
    global_p_w1 = torch.softmax(global_p1, dim=1)
    global_p_w2 = torch.softmax(global_p2, dim=1)

    # 1.3. Apply the weight map to each tensor (repaeat the weight map to the tensor size)
    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])
    tensor_f1 = global_p_w1 * tensor1
    tensor_f2 = global_p_w2 * tensor2

    # Step 2: Cross-Spatial Attention
    # 2.1. Calculate the spatial pooling for each tensor
    spatial1 = spatial_attention(tensor_f1, 'mean')
    spatial2 = spatial_attention(tensor_f2, 'mean')

    # 2.2. Calculate the weight map for each tensor 
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    # 2.3. Apply the weight map to each tensor
    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w2 * tensor1 + spatial_w1 * tensor2

    return tensor_f

'''
    Build the model for fusion with Parallel-Channel Attention and Cross-Spatial Attention
    Step 1: Parallel-Channel Attention apply for each tensor (def channel_attention -> softmax -> tensor_1 * weight_1; tensor_2 * weight_2)
    Step 2: Cross-Spatial Attention apply for each tensor (def spatial_attention -> softmax -> tensor_1 * weight_2 + tensor_2 * weight_1)
'''
def attention_fusion_weight_2(tensor1, tensor2, p_type):
    shape = tensor1.size()

    # Step 1: Parallel-Channel Attention
    # 1.1. Calculate the average global pooling for each tensor
    global_p1_avg = channel_attention(tensor1, pooling_type='attention_avg')
    global_p2_avg = channel_attention(tensor2, pooling_type='attention_avg')
    global_p_w1_avg = torch.softmax(global_p1_avg, dim=1)
    global_p_w2_avg = torch.softmax(global_p2_avg, dim=1)

    # 1.2. Calculate the max global pooling for each tensor
    global_p1_max = channel_attention(tensor1, pooling_type='attention_max')
    global_p2_max = channel_attention(tensor2, pooling_type='attention_max')
    global_p_w1_max = torch.softmax(global_p1_max, dim=1)
    global_p_w2_max = torch.softmax(global_p2_max, dim=1)

    # 1.3. Calculate mean of the weight map for each tensor
    global_p_w1 = (global_p_w1_avg + global_p_w2_max) / 2
    global_p_w2 = (global_p_w2_avg + global_p_w1_max) / 2

    # 1.3. Apply the weight map to each tensor (repaeat the weight map to the tensor size)
    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])
    tensor_f1 = global_p_w1 * tensor1
    tensor_f2 = global_p_w2 * tensor2

    # Step 2: Cross-Spatial Attention
    # 2.1. Calculate the spatial pooling for each tensor
    spatial1 = spatial_attention(tensor_f1, 'mean')
    spatial2 = spatial_attention(tensor_f2, 'mean')

    # 2.2. Calculate the weight map for each tensor 
    spatial_w1 = torch.softmax(torch.cat([spatial1, spatial2], dim=1), dim=1)[:, 0:1, :, :]
    spatial_w2 = torch.softmax(torch.cat([spatial1, spatial2], dim=1), dim=1)[:, 1:2, :, :]

    # 2.3. Apply the weight map to each tensor
    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f

def attention_fusion_weight(tensor1, tensor2, p_type):
    shape = tensor1.size()

    # Step 1: Parallel-Channel Attention
    subtract = torch.abs(tensor1 - tensor2)
    global_subtract = channel_attention(subtract, pooling_type='attention_avg')
    global_weight = torch.softmax(global_subtract, dim=1)
    global_weight = global_weight.repeat(1, 1, shape[2], shape[3])
    subtract_f = global_weight * subtract

    tensor_f1 = tensor1 + subtract_f
    tensor_f2 = tensor2 + subtract_f

    # Step 2: Cross-Spatial Attention
    # 2.1. Calculate the spatial pooling for each tensor
    spatial1 = spatial_attention(tensor_f1, 'mean')
    spatial2 = spatial_attention(tensor_f2, 'mean')

    # 2.2. Calculate the weight map for each tensor 
    spatial_w1 = torch.softmax(torch.cat([spatial1, spatial2], dim=1), dim=1)[:, 0:1, :, :]
    spatial_w2 = torch.softmax(torch.cat([spatial1, spatial2], dim=1), dim=1)[:, 1:2, :, :]

    # 2.3. Apply the weight map to each tensor
    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f

def attention_fusion_weight_3_2(tensor1, tensor2, p_type):
    shape = tensor1.size()

    # Step 1: Parallel-Channel Attention
    subtract = torch.abs(tensor1 - tensor2)
    global_subtract = channel_attention(subtract, pooling_type='attention_avg')
    global_weight = torch.softmax(global_subtract, dim=1)
    global_weight = global_weight.repeat(1, 1, shape[2], shape[3])
    subtract_f = global_weight * subtract

    tensor_f1 = tensor1 + subtract_f
    tensor_f2 = tensor2 + subtract_f

    # Step 2: Cross-Spatial Attention
    # 2.1. Calculate the spatial pooling for each tensor
    spatial1 = spatial_attention(tensor_f1, 'mean')
    spatial2 = spatial_attention(tensor_f2, 'mean')

    # 2.2. Calculate the weight map for each tensor 
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    # 2.3. Apply the weight map to each tensor
    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f

def attention_fusion_weight_4(tensor1, tensor2, p_type):
    shape = tensor1.size()

    # Step 1: Parallel-Channel Attention
    # 1.1. Calculate the average global pooling for each tensor
    global_p1_avg = channel_attention(tensor1, pooling_type='attention_avg')
    global_p2_avg = channel_attention(tensor2, pooling_type='attention_avg')
    global_w1 = torch.exp(global_p1_avg) / (torch.exp(global_p1_avg) + torch.exp(global_p2_avg) + EPSILON)
    global_w2 = torch.exp(global_p2_avg) / (torch.exp(global_p1_avg) + torch.exp(global_p2_avg) + EPSILON)

    # 1.3. Apply the weight map to each tensor (repaeat the weight map to the tensor size)
    global_w1 = global_w1.repeat(1, 1, shape[2], shape[3])
    global_w2 = global_w2.repeat(1, 1, shape[2], shape[3])
    tensor_channel = global_w1 * tensor1 + global_w2 * tensor2    

    # Step 2: Cross-Spatial Attention
    # 2.1. Calculate the spatial pooling for each tensor
    spatial1 = spatial_attention(tensor1, 'mean')
    spatial2 = spatial_attention(tensor2, 'mean')
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    # 2.3. Apply the weight map to each tensor
    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_spatial = spatial_w1 * tensor1 + spatial_w2 * tensor2

    tensor_f = (tensor_channel + tensor_spatial) / 2

    return tensor_f