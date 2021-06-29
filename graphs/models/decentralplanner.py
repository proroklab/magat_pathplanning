"""
An example for the model class
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphs.weights_initializer import weights_init
import numpy as np
import utils.graphUtils.graphML as gml
import utils.graphUtils.graphTools
from torchsummaryX import summary
from graphs.models.resnet_pytorch import *

class DecentralPlannerNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.S = None
        self.numAgents = self.config.num_agents
        # inW = self.config.map_w
        # inH = self.config.map_h

        inW = self.config.FOV + 2
        inH = self.config.FOV + 2
        # invW = 11
        # inH = 11

        convW = [inW]
        convH = [inH]
        numAction = 5

        use_vgg = False


        # ------------------ DCP v1.3  -  with maxpool + non stride in CNN
        if not self.config.use_dilated:
            numChannel = [3] + [32, 32, 64, 64, 128]
            numStride = [1, 1, 1, 1, 1]

            dimCompressMLP = 1
            numCompressFeatures = [self.config.numInputFeatures]

            nMaxPoolFilterTaps = 2
            numMaxPoolStride = 2
            # # 1 layer origin
            dimNodeSignals = [self.config.numInputFeatures]

        # # 2 layer - upsampling
        # dimNodeSignals = [256, self.config.numInputFeatures]

        # # 2 layer - down sampling
        # dimNodeSignals = [64, self.config.numInputFeatures]
        #
        # # 2 layer - down sampling -v2
        # dimNodeSignals = [64, 32]


        if self.config.use_dilated:
            # ------------------ DCP v5.1  -  using dilation CNN and less Maxpooling
            if self.config.use_dilated_version == 1:
                numChannel = [3] + [32, 32, 64, 64, 128]
                numStride = [1, 1, 1, 1, 1]
                numDilated = [1, 3, 1, 3, 1]
                nPaddingSzie = [1, 3, 1, 3, 1]
                #
                #
                dimCompressMLP = 1
                numCompressFeatures = [self.config.numInputFeatures]
                #
                nMaxPoolFilterTaps = 2
                numMaxPoolStride = 2
                dimNodeSignals = [self.config.numInputFeatures]

            # ------------------ DCP v5.2  -  using dilation CNN and less Maxpooling
            elif  self.config.use_dilated_version == 2:
                numChannel = [3] + [32, 32, 64, 64]
                numStride = [1, 1, 1, 1]
                numDilated = [1, 3, 1, 3]
                nPaddingSzie = [1, 3, 1, 3]

                dimCompressMLP = 1
                numCompressFeatures = [self.config.numInputFeatures]
                #
                nMaxPoolFilterTaps = 2
                numMaxPoolStride = 2
                dimNodeSignals = [self.config.numInputFeatures]

        # ------------------ DCP v2 - 1121
        # numChannel = [3] + [64, 64, 128, 128]
        # numStride = [1, 1, 2, 1]
        #
        #
        # dimCompressMLP = 3
        # numCompressFeatures = [2 ** 12, 2 ** 9, 128]

        # ------------------ DCP v3 - vgg
        # numChannel = [3] + [64, 128, 256, 256, 512, 512, 512, 512]
        # numStride = [1, 1, 2, 1, 1, 2, 1, 1]
        #
        # dimCompressMLP = 3
        # numCompressFeatures = [2 ** 12, 2 ** 12, 128]

        # ------------------ DCP v4 - vgg with max pool & dropout
        # use_vgg = True
        # cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        ## ------------------ GCN -------------------- ##
        # dimNodeSignals = [self.config.numInputFeatures]
        # nGraphFilterTaps = [self.config.nGraphFilterTaps,self.config.nGraphFilterTaps] # [2]
        nGraphFilterTaps = [self.config.nGraphFilterTaps]
        # --- actionMLP
        if self.config.use_dropout:
            dimActionMLP = 2
            numActionFeatures = [self.config.numInputFeatures, numAction]
        else:
            dimActionMLP = 1
            numActionFeatures = [numAction]


        #####################################################################
        #                                                                   #
        #                CNN to extract feature                             #
        #                                                                   #
        #####################################################################
        if use_vgg:
            self.ConvLayers = self.make_layers(cfg, batch_norm=True)
            self.compressMLP = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 128)
            )
            numCompressFeatures = [128]

        elif self.config.use_dilated:
            convl = []
            numConv = len(numChannel) - 1
            nFilterTaps = [3] * numConv

            for l in range(numConv):
                convl.append(nn.Conv2d(in_channels=numChannel[l], out_channels=numChannel[l+1],
                                       kernel_size=nFilterTaps[l], dilation =numDilated[l], stride=numStride[l], padding=nPaddingSzie[l], bias=True))
                convl.append(nn.BatchNorm2d(num_features=numChannel[l+1]))
                convl.append(nn.ReLU(inplace=True))

                W_tmp = int((convW[l] + 2*nPaddingSzie[l] - (nFilterTaps[l]-1)* numDilated[l] -1) / numStride[l]) + 1
                H_tmp = int((convH[l] + 2*nPaddingSzie[l] - (nFilterTaps[l]-1)* numDilated[l] -1) / numStride[l]) + 1
                # Adding maxpooling
                # if l == 1: #or l == 3:
                if l == 1 or l == 3:
                    convl.append(nn.MaxPool2d(kernel_size=nMaxPoolFilterTaps, stride=numMaxPoolStride))
                    W_tmp = int((W_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                    H_tmp = int((H_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                    # http://cs231n.github.io/convolutional-networks/
                convW.append(W_tmp)
                convH.append(H_tmp)

            self.ConvLayers = nn.Sequential(*convl)

            numFeatureMap = numChannel[-1] * convW[-1] * convH[-1]


            #####################################################################
            #                                                                   #
            #                MLP-feature compression                            #
            #                                                                   #
            #####################################################################

            numCompressFeatures = [numFeatureMap] + numCompressFeatures

            compressmlp = []
            for l in range(dimCompressMLP):
                compressmlp.append(nn.Linear(in_features=numCompressFeatures[l], out_features=numCompressFeatures[l+1], bias=True))
                compressmlp.append(nn.ReLU(inplace=True))

            self.compressMLP = nn.Sequential(*compressmlp)

        else:
            if self.config.CNN_mode == 'ResNetSlim_withMLP':
                convl = []
                convl.append(ResNetSlim(BasicBlock, [1, 1], out_map=False))
                convl.append(nn.Dropout(0.2))
                convl.append(nn.Flatten())
                convl.append(nn.Linear(in_features=1152, out_features=self.config.numInputFeatures, bias=True))
                self.ConvLayers = nn.Sequential(*convl)
                numFeatureMap = self.config.numInputFeatures
            elif self.config.CNN_mode == 'ResNetLarge_withMLP':
                convl = []
                convl.append(ResNet(BasicBlock, [1, 1, 1], out_map=False))
                convl.append(nn.Dropout(0.2))
                convl.append(nn.Flatten())
                convl.append(nn.Linear(in_features=1152, out_features=self.config.numInputFeatures, bias=True))
                self.ConvLayers = nn.Sequential(*convl)
                numFeatureMap = self.config.numInputFeatures
            elif self.config.CNN_mode == 'ResNetSlim':
                convl = []
                convl.append(ResNetSlim(BasicBlock, [1, 1], out_map=False))
                convl.append(nn.Dropout(0.2))
                self.ConvLayers = nn.Sequential(*convl)
                numFeatureMap = 1152
            elif self.config.CNN_mode == 'ResNetLarge':
                convl = []
                convl.append(ResNet(BasicBlock, [1, 1, 1], out_map=False))
                convl.append(nn.Dropout(0.2))
                self.ConvLayers = nn.Sequential(*convl)
                numFeatureMap = 1152
            else:
                convl = []
                numConv = len(numChannel) - 1
                nFilterTaps = [3] * numConv
                nPaddingSzie = [1] * numConv
                for l in range(numConv):
                    convl.append(nn.Conv2d(in_channels=numChannel[l], out_channels=numChannel[l + 1],
                                           kernel_size=nFilterTaps[l], stride=numStride[l], padding=nPaddingSzie[l],
                                           bias=True))
                    convl.append(nn.BatchNorm2d(num_features=numChannel[l + 1]))
                    convl.append(nn.ReLU(inplace=True))

                    # if self.config.use_dropout:
                    #     convl.append(nn.Dropout(p=0.2))
                    #     print('Dropout is add on CNN')

                    W_tmp = int((convW[l] - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
                    H_tmp = int((convH[l] - nFilterTaps[l] + 2 * nPaddingSzie[l]) / numStride[l]) + 1
                    # Adding maxpooling
                    if l % 2 == 0:
                        convl.append(nn.MaxPool2d(kernel_size=2))
                        W_tmp = int((W_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                        H_tmp = int((H_tmp - nMaxPoolFilterTaps) / numMaxPoolStride) + 1
                        # http://cs231n.github.io/convolutional-networks/
                    convW.append(W_tmp)
                    convH.append(H_tmp)

                self.ConvLayers = nn.Sequential(*convl)

                numFeatureMap = numChannel[-1] * convW[-1] * convH[-1]

            #####################################################################
            #                                                                   #
            #                MLP-feature compression                            #
            #                                                                   #
            #####################################################################



            numCompressFeatures = [numFeatureMap] + numCompressFeatures

            compressmlp = []
            for l in range(dimCompressMLP):
                compressmlp.append(
                    nn.Linear(in_features=numCompressFeatures[l], out_features=numCompressFeatures[l + 1], bias=True))
                compressmlp.append(nn.ReLU(inplace=True))
                # if self.config.use_dropout:
                #     compressmlp.append(nn.Dropout(p=0.2))
                #     print('Dropout is add on MLP')

            self.compressMLP = nn.Sequential(*compressmlp)

        self.numFeatures2Share = numCompressFeatures[-1]

        #####################################################################
        #                                                                   #
        #                    graph neural network                           #
        #                                                                   #
        #####################################################################

        self.L = len(nGraphFilterTaps)  # Number of graph filtering layers
        self.F = [numCompressFeatures[-1]] + dimNodeSignals  # Features
        # self.F = [numFeatureMap] + dimNodeSignals  # Features
        self.K = nGraphFilterTaps  # nFilterTaps # Filter taps
        self.E = 1  # Number of edge features
        self.bias = True

        gfl = []  # Graph Filtering Layers
        for l in range(self.L):
            # \\ Graph filtering stage:
            gfl.append(gml.GraphFilterBatch(self.F[l], self.F[l + 1], self.K[l], self.E, self.bias))
            # There is a 2*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.

            # \\ Nonlinearity
            if not self.config.no_ReLU:
                gfl.append(nn.ReLU(inplace=True))

        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl)  # Graph Filtering Layers

        #####################################################################
        #                                                                   #
        #                    MLP --- map to actions                         #
        #                                                                   #
        #####################################################################

        numActionFeatures = [self.F[-1]] + numActionFeatures
        actionsfc = []
        for l in range(dimActionMLP):
            if l < (dimActionMLP - 1):
                actionsfc.append(
                    nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))
                actionsfc.append(nn.ReLU(inplace=True))
            else:
                actionsfc.append(
                    nn.Linear(in_features=numActionFeatures[l], out_features=numActionFeatures[l + 1], bias=True))

            if self.config.use_dropout:
                actionsfc.append(nn.Dropout(p=0.2))
                print('Dropout is add on MLP')

        self.actionsMLP = nn.Sequential(*actionsfc)
        self.apply(weights_init)

    def make_layers(self, cfg, batch_norm=False):
        layers = []

        input_channel = 3
        for l in cfg:
            if l == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                continue

            layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

            if batch_norm:
                layers += [nn.BatchNorm2d(l)]

            layers += [nn.ReLU(inplace=True)]
            input_channel = l

        return nn.Sequential(*layers)


    def addGSO(self, S):

        # We add the GSO on real time, this GSO also depends on time and has
        # shape either B x N x N or B x E x N x N
        if self.E == 1:  # It is B x T x N x N
            assert len(S.shape) == 3
            self.S = S.unsqueeze(1)  # B x E x N x N
        else:
            assert len(S.shape) == 4
            assert S.shape[1] == self.E
            self.S = S
        # Remove nan data
        self.S[torch.isnan(self.S)] = 0
        if self.config.GSO_mode == 'dist_GSO_one':
            self.S[self.S > 0] = 1
        elif self.config.GSO_mode == 'full_GSO':
            self.S = torch.ones_like(self.S).to(self.config.device)
        # self.S[self.S > 0] = 1

    def forward(self, inputTensor):

        B = inputTensor.shape[0] # batch size
        # N = inputTensor.shape[1]
        # C =
        (B,N,C,W,H) = inputTensor.shape
        # print(inputTensor.shape)
        # print(B,N,C,W,H)
        # B x G x N

        input_currentAgent = inputTensor.reshape(B*N,C,W,H).to(self.config.device)

        featureMap = self.ConvLayers(input_currentAgent).to(self.config.device)

        featureMapFlatten = featureMap.view(featureMap.size(0), -1).to(self.config.device)

        compressfeature = self.compressMLP(featureMapFlatten).to(self.config.device)

        extractFeatureMap_old = compressfeature.reshape(B,N,self.numFeatures2Share).to(self.config.device)

        extractFeatureMap = extractFeatureMap_old.permute([0,2,1]).to(self.config.device)

        # DCP
        for l in range(self.L):
            # \\ Graph filtering stage:
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            self.GFL[2 * l].addGSO(self.S) # add GSO for GraphFilter

        # B x F x N - > B x G x N,
        sharedFeature = self.GFL(extractFeatureMap)

        (_, num_G, _) = sharedFeature.shape

        sharedFeature_permute =sharedFeature.permute([0,2,1]).to(self.config.device)
        sharedFeature_stack = sharedFeature_permute.reshape(B*N,num_G)

        action_predict = self.actionsMLP(sharedFeature_stack)

        return action_predict
