from functools import partial
import torch.nn as nn
import torch
# import pytorch_lightning as pl
import numpy as np



class BlockLayer(nn.Module):
    def __init__(self, num_blcks, block_layer, planes_in, planes_out, dropout=None, kernel_size=3, first_layer=False,
                 input_size=None):
        super(BlockLayer, self).__init__()

        self.blocks = nn.ModuleList()
        for i in range(num_blcks):
            if i == 0:
                self.blocks.append(block_layer(planes_in, planes_out, kernel_size=kernel_size, first_layer=first_layer,
                                               input_size=input_size))
            else:
                self.blocks.append(block_layer(planes_in, planes_out, kernel_size=kernel_size, first_layer=False,
                                               input_size=input_size))
            planes_in = planes_out
        if dropout is not None:
            self.dropout = nn.Dropout3d(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 0 and self.dropout is not None:
                x = self.dropout(x)
        return x



class ResidualBlock(nn.Module):
    def __init__(self, planes_in, planes_out, stride=1, kernel_size=3, first_layer=False, input_size=128):
        super(ResidualBlock, self).__init__()

        if dilated:
            self.conv1 = ConvolutionalBlock(planes_in=planes_in, planes_out=planes_out, first_layer=first_layer,
                                            kernel_size=kernel_size, dilation=2,
                                            activation=nn.ReLU, input_size=input_size)
            self.conv2 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                            kernel_size=1, dilation=1,
                                            activation=nn.ReLU, input_size=input_size)
        else:
            self.conv1 = ConvolutionalBlock(planes_in=planes_in, planes_out=planes_out, first_layer=first_layer,
                                            kernel_size=kernel_size, dilation=1,
                                            activation=nn.ReLU, input_size=input_size)
            self.conv2 = ConvolutionalBlock(planes_in=planes_out, planes_out=planes_out, first_layer=False,
                                            kernel_size=1,
                                            dilation=1, activation=nn.ReLU, input_size=input_size)
        if planes_in != planes_out:
            if dilated:
                self.sample = nn.Conv3d(planes_in, planes_out, (1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1),
                                        bias=False)  #
            else:
                self.sample = nn.Conv3d(planes_in, planes_out, (1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1),
                                        bias=False)  #


        else:
            self.sample = None
        # self.tanh = nn.Tanh()

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.conv2(x)

        if self.sample is not None:
            identity = self.sample(identity)

        x += identity

        return x


class EncoderLayer(nn.Module):
    def __init__(self, in_channel, base_inc_channel=16, layer_blocks=None, layer=BlockLayer, block=None,
                 feature_dilation=2, downsampling_stride=2, dropout=None, layer_widths=None, kernel_size=3):
        super(EncoderLayer, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        self.downsampling_zarib = []
        in_channel_layer = in_channel
        input_size = 128
        for i, num_blcks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_channel_layer = layer_widths[i]
            else:
                out_channel_layer = base_inc_channel * (feature_dilation ** (i))
            if dropout and i == 0:
                layer_dropout = dropout

            else:
                layer_dropout = None
            if i == 0:
                first_layer = True
            else:
                first_layer = False
            self.layers.append(layer(num_blcks=num_blcks, block_layer=block,
                                     planes_in=in_channel_layer, planes_out=out_channel_layer,
                                     dropout=layer_dropout, kernel_size=kernel_size,
                                     first_layer=first_layer, input_size=input_size))
            if i != len(layer_blocks) - 1:

                maxpool3d_layer = nn.MaxPool3d(kernel_size=(4, 4, 4),
                                               stride=(4, 4, 4), padding=0)
                self.downsampling_convolutions.append(maxpool3d_layer)

                input_size = input_size // 4
            print("Encoder {}:".format(i), in_channel_layer, out_channel_layer)
            in_channel_layer = out_channel_layer
        self.out_channel_layer = in_channel_layer


class ConvolutionalBlock(nn.Module):
    def __init__(self, planes_in, planes_out, first_layer=False, kernel_size=3, dilation=1, activation=None,
                 input_size=None):
        super(ConvolutionalBlock, self).__init__()
        if dilation == 1:
            padding = kernel_size // 2  # constant size
        else:
            # (In + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1
            if kernel_size == 3:
                if dilation == 2:
                    padding = 2
                elif dilation == 4:
                    padding = 4
                elif dilation == 3:
                    padding = 3
                else:
                    padding = None
            elif kernel_size == 1:
                padding = 0
        
        nrm = nn.LayerNorm([input_size, input_size, input_size])
        

        if first_layer:
            self.conv = nn.Sequential(*[activation(),
                                        nn.Conv3d(planes_in, planes_out, (kernel_size, kernel_size, kernel_size),
                                                  padding=padding, bias=False,
                                                  dilation=(dilation, dilation, dilation)),

                                        ])
        else:
            if activation is not None:
                if nrm is None:
                    self.conv = nn.Sequential(*[activation(),
                                                nn.Conv3d(planes_in, planes_out,
                                                          (kernel_size, kernel_size, kernel_size), padding=padding,
                                                          bias=False,
                                                          dilation=(dilation, dilation, dilation)),

                                                ])
                else:
                    self.conv = nn.Sequential(*[nrm, activation(),
                                                nn.Conv3d(planes_in, planes_out,
                                                          (kernel_size, kernel_size, kernel_size), padding=padding,
                                                          bias=False,
                                                          dilation=(dilation, dilation, dilation)),

                                                ])

            else:
                if nrm is None:
                    self.conv = nn.Sequential(*[
                        nn.Conv3d(planes_in, planes_out, (kernel_size, kernel_size, kernel_size), padding=padding,
                                  bias=False,
                                  dilation=(dilation, dilation, dilation)),

                    ])
                else:
                    self.conv = nn.Sequential(*[nrm, nn.Conv3d(planes_in, planes_out,
                                                               (kernel_size, kernel_size, kernel_size), padding=padding,
                                                               bias=False,
                                                               dilation=(dilation, dilation, dilation)),

                                                ])



    def forward(self, x):

        x = self.conv(x)


        return x


class AutoEncoder(nn.Module):
    def __init__(self, input_shape=None, in_channel=1, base_inc_channel=16, encoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2,
                 encoder_comp=EncoderLayer, layer_widths=None, block=None,
                 kernel_size=3,
                 dropout=0.5):
        super(AutoEncoder, self).__init__()
        self.base_inc_channel = base_inc_channel
        self.num_clusters = num_clusters
        self.Centers = nn.Parameter(data=torch.from_numpy(np.linspace(-1, 1, self.num_clusters)), requires_grad=True)
        self.first_center = True

        self.rep = 0
        self.fuzziness = 2

        if encoder_blocks is None:
            encoder_blocks = [1, 1, 1]


        inblock = 16
        if dilated:
            # (In + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1
            self.before_encoder = nn.Conv3d(1, inblock, kernel_size=(7, 7, 7),
                                            stride=(1, 1, 1), padding=7 // 2,
                                            bias=True, dilation=1)
        else:
            self.before_encoder = nn.Conv3d(1, inblock, kernel_size=(7, 7, 7),
                                            stride=(1, 1, 1), padding=7 // 2,
                                            bias=True, dilation=1)
        self.encoder = encoder_comp(in_channel=inblock, base_inc_channel=base_inc_channel, layer_blocks=encoder_blocks,
                                     block=block,
                                     feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                                     layer_widths=layer_widths, kernel_size=kernel_size, dropout=dropout)

        in_channel = self.encoder.out_channel_layer
        

        self.after_encoder = nn.Sequential(*[nn.LayerNorm([8, 8, 8]), nn.ReLU()])
        self.set_final_convolution(in_channel)
        self.relu = nn.ReLU()
        self.normlayer = nn.Sequential(
            # nn.BatchNorm3d(1),
            nn.Tanh()
        )
        output_dim = 1

        inp_dim = 8 * 8 * 8
       
        total_units = inp_dim
        self.batchnorm_linear1 = nn.LayerNorm(total_units)
        self.tanh = nn.Tanh()
       
        self.im_reg = nn.Sequential(
            *[nn.Linear(total_units, total_units, bias=False), nn.LayerNorm(total_units)])

        self.activation_layer = nn.Tanh()
        self.dropout = nn.Dropout(0)

        self.final_layer1 = nn.Linear(self.num_clusters + total_units, output_dim)


    def set_final_convolution(self, n_outputs, planes=64, kernel_size=1):

        self.final_convolution = nn.Conv3d(n_outputs, 1, kernel_size=(kernel_size, kernel_size, kernel_size),
                                           stride=(1, 1, 1), bias=False, padding=kernel_size // 2)
        if self.num_clusters > 0:
            self.conv_after_fcm = nn.Conv2d(self.num_clusters, 1, 1)


    def forward(self, y):
        x = self.before_encoder(y)
        x = self.encoder(x)

        x = self.after_encoder(x[0])

        z = self.final_convolution(x)

        x = z.flatten(-4, -1)  # flatten

        cat_reg = x
        cat_reg = self.im_reg(cat_reg)  # regression

        cat_reg = self.dropout(self.activation_layer(cat_reg))
        
        el = -2. / (self.fuzziness - 1)

        if self.num_clusters > 0:
            numerator = torch.zeros((*z.shape, self.num_clusters)).to(cat_reg.device)
            for i in range(self.num_clusters):
                numerator[..., i] = torch.pow((z - self.Centers[i]) + 0.0001, el)
            numerator = self.activation_layer(numerator)

            out = torch.concat([numerator.flatten(-5, -2).mean(-2), cat_reg], 1)
        else:
            out = cat_reg
        
        if len(out.size()) == 1:
            out = out.reshape(1, -1)
        y = self.final_layer1(out)

        return y


class Encoder(EncoderLayer):
    def forward(self, x):
        outputs = list()
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        outputs.insert(0, x)
        return outputs


class fuzzydml(AutoEncoder):
    def __init__(self, *args, encoder_comp=Encoder, **kwargs):
        global dilated
        dilated = args[0].dilated
        global num_clusters
        num_clusters = args[0].nCluster
        dropout = 0
        global block_used
        block = ResidualBlock
        block_used = 'residual'
        super().__init__(*args, encoder_comp=encoder_comp, dropout=dropout,
                         block=block, **kwargs)
        self.netName = 'fuzzydml'
