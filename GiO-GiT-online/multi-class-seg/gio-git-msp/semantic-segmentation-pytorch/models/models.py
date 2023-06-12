import torch
import torch.nn as nn
import torchvision

from . import resnet, resnext, mobilenet, hrnet
from lib.nn import SynchronizedBatchNorm2d
from torchsphere import nn as spherenn
from torchsphere.nn import AngleLoss
from torchhyp import nn as hypnn
from distance import Logits, SoftTarget
BatchNorm2d = SynchronizedBatchNorm2d


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, net_enc, net_dec, crit, deep_sup_scale=None):
        super(SegmentationModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.crit = crit
        self.deep_sup_scale = deep_sup_scale

    def forward(self, feed_dict, *, segSize=None):
        # training
        if segSize is None:
            if self.deep_sup_scale is not None: # use deep supervision technique
                (pred, pred_deepsup) = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))
            else:
                pred, pred_s, pred_h = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True))

            loss_e = self.crit(pred, feed_dict['seg_label'])
            loss_s = self.crit(pred_s, feed_dict['seg_label'])
            loss_h = self.crit(pred_h, feed_dict['seg_label'])
            loss = loss_e + 0.1*loss_s + 0.1*loss_h

            if self.deep_sup_scale is not None:
                loss_deepsup = self.crit(pred_deepsup, feed_dict['seg_label'])
                loss = loss + loss_deepsup * self.deep_sup_scale

            acc = self.pixel_acc(pred, feed_dict['seg_label'])
            return loss, acc
        # inference
        else:
            pred, pred_s, pred_h = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), segSize=segSize)
            return pred, pred_s, pred_h


class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        elif arch == 'resnet18':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet18dilated':
            orig_resnet = resnet.__dict__['resnet18'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet34':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet34dilated':
            raise NotImplementedError
            orig_resnet = resnet.__dict__['resnet34'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnext101':
            orig_resnext = resnext.__dict__['resnext101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnext) # we can still use class Resnet
        elif arch == 'hrnetv2':
            net_encoder = hrnet.__dict__['hrnetv2'](pretrained=pretrained)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='ppm_deepsup',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False,
                      curvature_s=1.0, curvature_h=1.0):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(num_class=num_class,
                                    fc_dim=fc_dim,
                                    use_softmax=use_softmax)
        elif arch == 'c1':
            net_decoder = C1(num_class=num_class,
                             fc_dim=fc_dim,
                             use_softmax=use_softmax)
        elif arch == 'ppm':
            net_decoder = PPM(num_class=num_class,
                              fc_dim=fc_dim,
                              use_softmax=use_softmax)
        elif arch == 'ppm_deepsup':
            net_decoder = PPMDeepsup(num_class=num_class,
                                     fc_dim=fc_dim,
                                     use_softmax=use_softmax)
        elif arch == 'upernet_lite':
            net_decoder = UPerNet(num_class=num_class,
                                  fc_dim=fc_dim,
                                  use_softmax=use_softmax,
                                  fpn_dim=256)
        elif arch == 'upernet':
            net_decoder = UPerNet(num_class=num_class,
                                  fc_dim=fc_dim,
                                  use_softmax=use_softmax,
                                  fpn_dim=512)

        ######################################################
        elif arch == 'upernet_sphere':
            net_decoder = UPerNet_Sphere(num_class=num_class,
                                         fc_dim=fc_dim,
                                         use_softmax=use_softmax,
                                         fpn_dim=512,
                                         curvature_s=curvature_s)

        elif arch == 'upernet_hyper':
            net_decoder = UPerNet_Hyper(num_class=num_class,
                                        fc_dim=fc_dim,
                                        use_softmax=use_softmax,
                                        fpn_dim=512,
                                        curvature_h=curvature_h)

        elif arch == 'upernet_mix':
            net_decoder = UPerNet_Mix(num_class=num_class,
                                      fc_dim=fc_dim,
                                      use_softmax=use_softmax,
                                      fpn_dim=512,
                                      curvature_s=curvature_s,
                                      curvature_h=curvature_h)
        ######################################################

        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False),
                         BatchNorm2d(out_planes),
                         nn.ReLU(inplace=True))


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x); conv_out.append(x);
        x = self.layer2(x); conv_out.append(x);
        x = self.layer3(x); conv_out.append(x);
        x = self.layer4(x); conv_out.append(x);

        if return_feature_maps:
            return conv_out
        return [x]


class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif dilate_scale == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        if return_feature_maps:
            conv_out = []
            for i in range(self.total_idx):
                x = self.features[i](x)
                if i in self.down_idx:
                    conv_out.append(x)
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1DeepSup, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)

        if self.use_softmax: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x


# pyramid pooling
class PPM(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# pyramid pooling, deep supervision
class PPMDeepsup(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6)):
        super(PPMDeepsup, self).__init__()
        self.use_softmax = use_softmax

        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_class, kernel_size=1)
        )
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.dropout_deepsup = nn.Dropout2d(0.1)

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        # deep sup
        conv4 = conv_out[-2]
        _ = self.cbr_deepsup(conv4)
        _ = self.dropout_deepsup(_)
        _ = self.conv_last_deepsup(_)

        x = nn.functional.log_softmax(x, dim=1)
        _ = nn.functional.log_softmax(_, dim=1)

        return (x, _)


# upernet
class UPerNet(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256):
        super(UPerNet, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                                               BatchNorm2d(512),
                                               nn.ReLU(inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv    = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                                             BatchNorm2d(fpn_dim),
                                             nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
                                       nn.Conv2d(fpn_dim, num_class, kernel_size=1))

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(pool_scale(conv5),
                                                               (input_size[2], input_size[3]),
                                                               mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(fpn_feature_list[i],
                                                         output_size,
                                                         mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:       # is True during inference
            x = nn.functional.interpolate(x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x


################
# upernet_sphere
class UPerNet_Sphere(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256,
                 curvature_s=1.0):
        super(UPerNet_Sphere, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                                               BatchNorm2d(512),
                                               nn.ReLU(inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv    = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                                             BatchNorm2d(fpn_dim),
                                             nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        # Euclidean layer
        self.conv_last = nn.Sequential(conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
                                       nn.Conv2d(fpn_dim, num_class, kernel_size=1))

        # spherical layer
        self.num_class  = num_class
        self.conv3x3    = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)
        self.maxpool    = nn.MaxPool2d(4, stride=4)
        self.upsample   = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.fc_s       = spherenn.AngleLinear(fpn_dim, num_class, curvature=curvature_s, m=1)
        self.KL_distance = SoftTarget(T=1.0)

    def get_score_sphere(self, x_e, x_s):
        batch_size = x_e.shape[0]
        num_class  = x_e.shape[1]
        H = x_e.shape[2]
        W = x_e.shape[3]

        scores = torch.zeros([batch_size, H, W])
        for b in range(batch_size):
           for h in range(H):
               for w in range(W):
                   scores_ele = self.KL_distance(x_s[b, :, h, w].unsqueeze(0), 
                                                 x_e[b, :, h, w].unsqueeze(0))
                   # embedding: torch.tanh(scores_ele); logit: 1-torch.tanh(scores_ele)
                   scores[b, h, w] = 1-torch.tanh(scores_ele) 
        return scores

    def forward_sphere(self, x):
        x = self.maxpool(x)

        batch_size  = x.shape[0]
        num_channel = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
 
        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, (batch_size*H*W, num_channel))
        x = self.fc_s(x)
        x = x[0]
        x = torch.reshape(x, (batch_size, H, W, self.num_class))
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)

        return x

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(pool_scale(conv5),
                                                               (input_size[2], input_size[3]),
                                                               mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(fpn_feature_list[i],
                                                         output_size,
                                                         mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)

        # Euclidean layer
        x_e = self.conv_last(fusion_out)

        # spherical layer
        x_s = self.conv3x3(fusion_out)
        x_s = self.forward_sphere(x_s)

        if self.use_softmax:       # is True during inference
            x_s = self.get_score_sphere(x_e, x_s).unsqueeze(1)

            x_e = nn.functional.interpolate(x_e, size=segSize, mode='bilinear', align_corners=False)
            x_s = nn.functional.interpolate(x_s, size=segSize, mode='bilinear', align_corners=False)

            x_e = nn.functional.softmax(x_e, dim=1)

            return x_e, x_s.cuda()

        x_e = nn.functional.log_softmax(x_e, dim=1)
        x_s = nn.functional.log_softmax(x_s, dim=1)

        return x_e, x_s, torch.zeros_like(x_s)


# upernet_hyper
class UPerNet_Hyper(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256,
                 curvature_h=1.0):
        super(UPerNet_Hyper, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                                               BatchNorm2d(512),
                                               nn.ReLU(inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv    = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                                             BatchNorm2d(fpn_dim),
                                             nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        # Euclidean layer
        self.conv_last = nn.Sequential(conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
                                       nn.Conv2d(fpn_dim, num_class, kernel_size=1))

        # hyperbolic layer
        self.num_class  = num_class
        self.conv3x3    = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)
        self.maxpool    = nn.MaxPool2d(4, stride=4)
        self.upsample   = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.tp         = hypnn.ToPoincare(c=curvature_h, train_x=False, train_c=False, ball_dim=fpn_dim)
        self.mlr        = hypnn.HyperbolicMLR(ball_dim=fpn_dim, n_classes=num_class, c=curvature_h)
        self.KL_distance = SoftTarget(T=1.0)

    def get_score_hyper(self, x_e, x_h):
        batch_size = x_e.shape[0]
        num_class  = x_e.shape[1]
        H = x_e.shape[2]
        W = x_e.shape[3]

        scores = torch.zeros([batch_size, H, W])
        for b in range(batch_size):
           for h in range(H):
               for w in range(W):
                   scores_ele = self.KL_distance(x_h[b, :, h, w].unsqueeze(0), 
                                                 x_e[b, :, h, w].unsqueeze(0))
                   # embedding: torch.tanh(scores_ele); logit: 1-torch.tanh(scores_ele)
                   scores[b, h, w] = 1-torch.tanh(scores_ele)
        return scores

    def forward_hyper(self, x):
        x = self.maxpool(x)

        batch_size  = x.shape[0]
        num_channel = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
 
        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, (batch_size*H*W, num_channel))
        x = self.tp(x)
        x = self.mlr(x)
        x = torch.reshape(x, (batch_size, H, W, self.num_class))
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)

        return x

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(pool_scale(conv5),
                                                               (input_size[2], input_size[3]),
                                                               mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(fpn_feature_list[i],
                                                         output_size,
                                                         mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)

        # Euclidean layer
        x_e = self.conv_last(fusion_out)

        # hyperbolic layer
        x_h = self.conv3x3(fusion_out)
        x_h = self.forward_hyper(x_h)

        if self.use_softmax:       # is True during inference
            x_h = self.get_score_hyper(x_e, x_h).unsqueeze(1)

            x_e = nn.functional.interpolate(x_e, size=segSize, mode='bilinear', align_corners=False)
            x_h = nn.functional.interpolate(x_h, size=segSize, mode='bilinear', align_corners=False)

            x_e = nn.functional.softmax(x_e, dim=1)

            return x_e, x_h.cuda()

        x_e = nn.functional.log_softmax(x_e, dim=1)
        x_h = nn.functional.log_softmax(x_h, dim=1)

        return x_e, torch.zeros_like(x_h), x_h


# upernet_mix
class UPerNet_Mix(nn.Module):
    def __init__(self, num_class=150, fc_dim=4096,
                 use_softmax=False, pool_scales=(1, 2, 3, 6),
                 fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=256,
                 curvature_s=1.0, curvature_h=1.0):
        super(UPerNet_Mix, self).__init__()
        self.use_softmax = use_softmax

        # PPM Module
        self.ppm_pooling = []
        self.ppm_conv = []

        for scale in pool_scales:
            self.ppm_pooling.append(nn.AdaptiveAvgPool2d(scale))
            self.ppm_conv.append(nn.Sequential(nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                                               BatchNorm2d(512),
                                               nn.ReLU(inplace=True)))
        self.ppm_pooling = nn.ModuleList(self.ppm_pooling)
        self.ppm_conv    = nn.ModuleList(self.ppm_conv)
        self.ppm_last_conv = conv3x3_bn_relu(fc_dim + len(pool_scales)*512, fpn_dim, 1)

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                                             BatchNorm2d(fpn_dim),
                                             nn.ReLU(inplace=True)))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for i in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        # Euclidean layer
        self.conv_last = nn.Sequential(conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
                                       nn.Conv2d(fpn_dim, num_class, kernel_size=1))

        # spherical layer
        self.num_class  = num_class
        self.conv3x3    = conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1)
        self.maxpool    = nn.MaxPool2d(4, stride=4)
        self.upsample   = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.fc_s       = spherenn.AngleLinear(fpn_dim, num_class, curvature=curvature_s, m=1)
        self.KL_distance = SoftTarget(T=1.0)

        # hyperbolic layer
        self.tp         = hypnn.ToPoincare(c=curvature_h, train_x=False, train_c=False, ball_dim=fpn_dim)
        self.mlr        = hypnn.HyperbolicMLR(ball_dim=fpn_dim, n_classes=num_class, c=curvature_h)

    def get_score_sphere(self, x_e, x_s):
        batch_size = x_e.shape[0]
        num_class  = x_e.shape[1]
        H = x_e.shape[2]
        W = x_e.shape[3]

        scores = torch.zeros([batch_size, H, W])
        for b in range(batch_size):
           for h in range(H):
               for w in range(W):
                   scores_ele = self.KL_distance(x_s[b, :, h, w].unsqueeze(0), 
                                                 x_e[b, :, h, w].unsqueeze(0))
                   # embedding: torch.tanh(scores_ele); logit: 1-torch.tanh(scores_ele)
                   scores[b, h, w] = 1-torch.tanh(scores_ele)
        return scores

    def get_score_hyper(self, x_e, x_h):
        batch_size = x_e.shape[0]
        num_class  = x_e.shape[1]
        H = x_e.shape[2]
        W = x_e.shape[3]

        scores = torch.zeros([batch_size, H, W])
        for b in range(batch_size):
           for h in range(H):
               for w in range(W):
                   scores_ele = self.KL_distance(x_h[b, :, h, w].unsqueeze(0), 
                                                 x_e[b, :, h, w].unsqueeze(0))
                   # embedding: torch.tanh(scores_ele); logit: 1-torch.tanh(scores_ele)
                   scores[b, h, w] = 1-torch.tanh(scores_ele)
        return scores

    def forward_sphere(self, x):
        x = self.maxpool(x)

        batch_size  = x.shape[0]
        num_channel = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
 
        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, (batch_size*H*W, num_channel))
        x = self.fc_s(x)
        x = x[0]
        x = torch.reshape(x, (batch_size, H, W, self.num_class))
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)

        return x

    def forward_hyper(self, x):
        x = self.maxpool(x)

        batch_size  = x.shape[0]
        num_channel = x.shape[1]
        H = x.shape[2]
        W = x.shape[3]
 
        x = x.permute(0, 2, 3, 1)
        x = torch.reshape(x, (batch_size*H*W, num_channel))
        x = self.tp(x)
        x = self.mlr(x)
        x = torch.reshape(x, (batch_size, H, W, self.num_class))
        x = x.permute(0, 3, 1, 2)
        x = self.upsample(x)

        return x

    def forward(self, conv_out, segSize=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale, pool_conv in zip(self.ppm_pooling, self.ppm_conv):
            ppm_out.append(pool_conv(nn.functional.interpolate(pool_scale(conv5),
                                                               (input_size[2], input_size[3]),
                                                               mode='bilinear', align_corners=False)))
        ppm_out = torch.cat(ppm_out, 1)
        f = self.ppm_last_conv(ppm_out)

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x) # lateral branch

            f = nn.functional.interpolate(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False) # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse() # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(fpn_feature_list[i],
                                                         output_size,
                                                         mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)

        # Euclidean layer
        x_e = self.conv_last(fusion_out)

        # spherical layer
        x_s = self.conv3x3(fusion_out)
        x_s = self.forward_sphere(x_s)

        # hyperbolic layer
        x_h = self.conv3x3(fusion_out)
        x_h = self.forward_hyper(x_h)

        if self.use_softmax:       # is True during inference
            x_s = self.get_score_sphere(x_e, x_s).unsqueeze(1)
            x_h = self.get_score_hyper(x_e, x_h).unsqueeze(1)

            x_e = nn.functional.interpolate(x_e, size=segSize, mode='bilinear', align_corners=False)
            x_s = nn.functional.interpolate(x_s, size=segSize, mode='bilinear', align_corners=False)
            x_h = nn.functional.interpolate(x_h, size=segSize, mode='bilinear', align_corners=False)

            x_e = nn.functional.softmax(x_e, dim=1)

            return x_e, x_s.cuda(), x_h.cuda()

        x_e = nn.functional.log_softmax(x_e, dim=1)
        x_s = nn.functional.log_softmax(x_s, dim=1)
        x_h = nn.functional.log_softmax(x_h, dim=1)

        return x_e, x_s, x_h
############################
