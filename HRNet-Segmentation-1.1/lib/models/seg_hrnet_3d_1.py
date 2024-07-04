# ------------------------------------------------------------------------------
# 我只是将hrnet由原来的conv2d改为conv3d
# 为了减少计算量,我将每个stage的4个block改为2个block
# 如果有必要,还需要再将每个block的NUM_CHANNELS改的更小.
# 在最后一个stage的每个branch添加一个tam_block,在时间维度添加了注意力.
# 从头到尾,不改变帧数(也就是时间数).只是在最后变为2维图像,直接抛弃帧数.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import functools

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BatchNorm3d = nn.BatchNorm3d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=(1, 1, 1)):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1, 1), downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), bias=False)
        # 一维卷积,输出通道数不变,输出尺寸不变
        self.bn1 = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1), bias=False)
        # kernel_size=3,stride=1,padding=1,则N=(W-3+2)/1+1=w,所以输出尺寸不变.输出通道数不变
        self.bn2 = BatchNorm3d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=(1, 1, 1), bias=False)
        # 一维卷积,输出通道数变为planes * 4,输出尺寸不变
        self.bn3 = BatchNorm3d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method,
                 multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks, num_inchannels, num_channels):
        # num_branches即为branches分支数,必然要和num_blocks列表,num_channels列表,num_inchannels列表对应.
        # 经过transition_layer操作后,输出的不同branch的通道数已经和后面的stage的不同branch的通道数相同了.
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=(1, 1, 1)):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion, kernel_size=(1, 1, 1), stride=stride,
                          bias=False),
                BatchNorm3d(num_channels[branch_index] * block.expansion, momentum=BN_MOMENTUM), )

        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(nn.Conv3d(num_inchannels[j], num_inchannels[i], (1, 1, 1), (1, 1, 1), (0, 0, 0),
                                                bias=False),
                                      BatchNorm3d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(nn.Conv3d(num_inchannels[j], num_outchannels_conv3x3,
                                                        (3, 3, 3), (1, 2, 2), (1, 1, 1), bias=False),
                                              BatchNorm3d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(nn.Conv3d(num_inchannels[j], num_outchannels_conv3x3,
                                                        (3, 3, 3), (1, 2, 2), (1, 1, 1), bias=False),
                                              BatchNorm3d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                              nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[4, height_output, width_output],
                                          mode='trilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class tam_block(nn.Module):
    def __init__(self, in_planes=4, ratio=4):
        super(tam_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc1 = nn.Conv3d(in_planes, in_planes // ratio, (1, 1, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_planes // ratio, in_planes, (1, 1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x1 = torch.transpose(x, 1, 2)
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(torch.transpose(x, 1, 2)))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(torch.transpose(x, 1, 2)))))
        # out = avg_out + max_out
        # out = self.sigmoid(torch.transpose(out, 1, 2))
        out = self.sigmoid(torch.transpose((avg_out + max_out), 1, 2))

        return torch.mul(x, out)


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        "stem net"
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)  # 图像尺寸缩小2倍
        self.conv1 = nn.Conv3d(4, 64, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.bn1 = BatchNorm3d(64, momentum=BN_MOMENTUM)
        # self.conv2 = nn.Conv3d(64, 64, kernel_size=(3,3,3), stride=(1,2,2), padding=(1,1,1), bias=False)
        # self.bn2 = BatchNorm3d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        "stage1"
        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]  # 64
        block = blocks_dict[self.stage1_cfg['BLOCK']]  # BOTTLENECK
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]  # 4
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        # self.layer1 = self._make_layer(block, 4, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        "stage2"
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        "stage3"
        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        "stage4"
        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output=True)

        "temporal attention"
        # self.tam_block = tam_block(in_planes=4, ratio=2)  # 时间维度2倍压缩
        self.tam_block = tam_block(in_planes=4, ratio=4)  # 时间维度4倍压缩

        "last_layer"
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        self.last_layer_1 = nn.Sequential(
            nn.Conv3d(in_channels=last_inp_channels, out_channels=last_inp_channels,
                      kernel_size=(4, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            BatchNorm3d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),)

        self.last_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=config.DATASET.NUM_CLASSES,
                      kernel_size=extra.FINAL_CONV_KERNEL, stride=1, padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            BatchNorm2d(config.DATASET.NUM_CLASSES, momentum=BN_MOMENTUM),
            nn.Sigmoid())

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):  # num_branches_cur
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv3d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                      (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False),
                            BatchNorm3d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(nn.Conv3d(inchannels, outchannels, (3, 3, 3), (1, 2, 2), (1, 1, 1), bias=False),
                                      BatchNorm3d(outchannels, momentum=BN_MOMENTUM),
                                      nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=(1, 1, 1)):
        downsample = None
        if stride != (1, 1, 1) or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes * block.expansion, kernel_size=(1, 1, 1), stride=stride, bias=False),
                BatchNorm3d(planes * block.expansion, momentum=BN_MOMENTUM), )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method,
                                     reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        "stem net"
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        # x = self.relu(x)
        "stage1"
        x = self.layer1(x)
        "transition1"
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        "stage2"
        y_list = self.stage2(x_list)
        "transition3"
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        "stage3"
        y_list = self.stage3(x_list)
        "transition4"
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        "stage4"
        x = self.stage4(x_list)

        "temporal attention"
        x0 = self.tam_block(x[0])
        x1 = self.tam_block(x[1])
        x2 = self.tam_block(x[2])
        x3 = self.tam_block(x[3])

        "Upsampling"
        x0_h, x0_w = x0.size(3), x0.size(4)
        # x1 = F.upsample(x1, size=(4, x0_h, x0_w), mode='trilinear')
        # nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
        x1 = F.interpolate(x1, size=(4, x0_h, x0_w), mode='trilinear', align_corners=True)
        # UserWarning: Default upsampling behavior when mode=trilinear is changed to align_corners=False since 0.4.0.
        # Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample
        # for details.
        # x2 = F.upsample(x2, size=(4, x0_h, x0_w), mode='trilinear')
        x2 = F.interpolate(x2, size=(4, x0_h, x0_w), mode='trilinear', align_corners=True)
        # x3 = F.upsample(x3, size=(4, x0_h, x0_w), mode='trilinear', )
        x3 = F.interpolate(x3, size=(4, x0_h, x0_w), mode='trilinear', align_corners=True)

        x = torch.cat([x[0], x1, x2, x3], 1)

        "last_layer"
        x = self.last_layer_1(x)
        x = torch.squeeze(x)
        x = self.last_layer_2(x)

        return x

    def init_weights(self, pretrained='', ):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            # for k, _ in pretrained_dict.items():
            #    logger.info(
            #        '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_seg_model(cfg, **kwargs):
    model = HighResolutionNet(cfg, **kwargs)
    # model.init_weights(cfg.MODEL.PRETRAINED)

    return model