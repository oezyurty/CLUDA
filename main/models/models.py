import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F


# from utils import weights_init

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################

########## CNN #############################
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

        # weights_init(self.conv_block1)
        # weights_init(self.conv_block2)
        # weights_init(self.conv_block3)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions


########## TCN #############################
torch.backends.cudnn.benchmark = True  # might be required to fasten TCN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TCN(nn.Module):
    def __init__(self, configs):
        super(TCN, self).__init__()

        in_channels0 = configs.input_channels
        out_channels0 = configs.tcn_layers[1]
        kernel_size = configs.tcn_kernel_size
        stride = 1
        dilation0 = 1
        padding0 = (kernel_size - 1) * dilation0

        self.net0 = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
            weight_norm(nn.Conv1d(out_channels0, out_channels0, kernel_size, stride=stride, padding=padding0,
                                  dilation=dilation0)),
            nn.ReLU(),
        )

        self.downsample0 = nn.Conv1d(in_channels0, out_channels0, 1) if in_channels0 != out_channels0 else None
        self.relu = nn.ReLU()

        in_channels1 = configs.tcn_layers[0]
        out_channels1 = configs.tcn_layers[1]
        dilation1 = 2
        padding1 = (kernel_size - 1) * dilation1
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
            nn.Conv1d(out_channels1, out_channels1, kernel_size, stride=stride, padding=padding1, dilation=dilation1),
            nn.ReLU(),
        )
        self.downsample1 = nn.Conv1d(out_channels1, out_channels1, 1) if in_channels1 != out_channels1 else None

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False, padding=padding0,
                      dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),

            nn.Conv1d(out_channels0, out_channels0, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding0, dilation=dilation0),
            Chomp1d(padding0),
            nn.BatchNorm1d(out_channels0),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(out_channels0, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),

            nn.Conv1d(out_channels1, out_channels1, kernel_size=kernel_size, stride=stride, bias=False,
                      padding=padding1, dilation=dilation1),
            Chomp1d(padding1),
            nn.BatchNorm1d(out_channels1),
            nn.ReLU(),
        )

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x0 = self.conv_block1(inputs)
        res0 = inputs if self.downsample0 is None else self.downsample0(inputs)
        out_0 = self.relu(x0 + res0)

        x1 = self.conv_block2(out_0)
        res1 = out_0 if self.downsample1 is None else self.downsample1(out_0)
        out_1 = self.relu(x1 + res1)

        out = out_1[:, :, -1]
        return out


######## RESNET ##############################################
class RESNET18(nn.Module):
    def __init__(self, configs):
        layers = [2, 2, 2, 2]
        block = BasicBlock
        self.inplanes = configs.input_channels
        super(RESNET18, self).__init__()
        self.layer1 = self._make_layer(block, configs.mid_channels, layers[0], stride=configs.stride)
        self.layer2 = self._make_layer(block, configs.mid_channels * 2, layers[1], stride=1)
        self.layer3 = self._make_layer(block, configs.final_out_channels, layers[2], stride=1)
        self.layer4 = self._make_layer(block, configs.final_out_channels, layers[3], stride=1)

        self.avgpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adaptive_pool(x)

        x_flat = x.reshape(x.shape[0], -1)
        return x_flat


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = F.relu(out)

        return out

#Helper for TS-SASA

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input):
        """Forward function.
        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size
        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor
        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=input.device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

#TS-SASA model (featured in AAAI' 21)
class SASA(nn.Module):
    def __init__(self, x_dim=41, h_dim=2, n_segments=5, seg_length=4, temp=1.0):
        super(SASA, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_segments = n_segments
        self.seg_length = seg_length
        self.temp=temp

        self.att_query =  nn.Linear(h_dim, h_dim)
        self.att_key =  nn.Linear(h_dim, h_dim)
        self.att_value =  nn.Linear(h_dim, h_dim)

        self.rnn_list = nn.ModuleList([nn.LSTM(1, h_dim, batch_first=True) for i in range(x_dim)])

        self.sparsemax = Sparsemax()

        self.softmax = nn.Softmax(dim=1)




    def forward(self, x):
        #x: batch * length * num_channel

        len_ = x.shape[1]
        list_att_weights_intra = []
        list_att_weights_inter = []
        list_h_i = []
        list_Z_i = []
        list_H_i = []

        for i in range(0, x.shape[2]):
            #We iterate over each channel/measurement
            channel_x = x[:,:,i].unsqueeze(-1)

            channel_h_n_list = []

            for n in range(1,self.n_segments+1):
                window_x = channel_x[:,len_-n*self.seg_length:len_,:]
                _, (h_n, _) = self.rnn_list[i](window_x)

                channel_h_n_list.append(h_n)

            channel_h_n = torch.cat(channel_h_n_list, dim=0) # batch * num_seg, h_dim
            list_h_i.append(channel_h_n)

            Q = self.att_query(channel_h_n).reshape(x.shape[0], self.n_segments, -1) # batch, num_seg, h_dim
            K = self.att_key(channel_h_n).reshape(x.shape[0], self.n_segments, -1)
            V = self.att_value(channel_h_n).reshape(x.shape[0], self.n_segments, -1)

            att_weights_intra = self.self_attention_fn(Q, K) # batch, num_seg
            list_att_weights_intra.append(att_weights_intra)
            #Expand it so that one attention weight can multiply all the vector of size h_dim
            att_weights_intra_expanded = att_weights_intra.unsqueeze(-1).expand(att_weights_intra.shape[0], att_weights_intra.shape[1], V.shape[-1])

            Z_i = (att_weights_intra_expanded * V).mean(dim=1) #Batch * h_dim
            list_Z_i.append(Z_i)

        for i in range(0, x.shape[2]):
            Z_i = list_Z_i[i]

            other_h_i = [list_h_i[j] for j in range(0, x.shape[2]) if j!=i]
            other_h_i = torch.cat(other_h_i, dim=0) # batch * num_seg * (num_channel-1), h_dim
            other_h_i = other_h_i.reshape(x.shape[0], -1, Z_i.shape[-1])

            att_weights_inter = self.attention_fn(Z_i, other_h_i)
            list_att_weights_inter.append(att_weights_inter)

            #Now get the features for prediction
            att_weights_inter_expanded = att_weights_inter.unsqueeze(-1).expand(att_weights_inter.shape[0], att_weights_inter.shape[1], other_h_i.shape[-1])
            U_i = (att_weights_inter_expanded*other_h_i).mean(dim=1)

            H_i = torch.cat([Z_i, U_i], dim=-1)

            list_H_i.append(H_i)

        H = torch.cat(list_H_i, dim=1)

        return list_att_weights_intra, list_att_weights_inter, H


    def self_attention_fn(self, Q, K):
        #Get cosine similarity of Q and K for each element
        att_weight = torch.bmm(Q, K.transpose(2,1)) # Batch * num_seg * num_seg

        att_weight = att_weight.mean(dim=2) # Batch * num_seg

        att_weight /= math.sqrt(Q.shape[-1])

        att_weight /= self.temp

        att_weight = self.sparsemax(att_weight)
        #att_weight = self.softmax(att_weight)

        #print("Here is the attention weight of sample 0:")
        #print(att_weight[0,:])
        #print("Here is the attention weight of sample 10:")
        #print(att_weight[10,:])

        return att_weight

    def attention_fn(self, Q, K):
        #Get cosine similarity of Q and K for each element
        att_weight = torch.bmm(F.normalize(Q, dim=-1).reshape(Q.shape[0],1,Q.shape[1]), F.normalize(K, dim=-1).transpose(2,1)) # Batch, 1, num_seg * (num_channel - 1)

        att_weight = att_weight.mean(dim=1) # Batch,  num_seg * (num_channel - 1)

        #att_weight /= Q.shape[-1]

        att_weight /= self.temp

        att_weight = self.sparsemax(att_weight)
        #att_weight = self.softmax(att_weight)

        #print("Here is the attention weight of sample 0:")
        #print(att_weight[0,:])
        #print("Here is the attention weight of sample 10:")
        #print(att_weight[10,:])

        #quit()

        return att_weight





#VRNN: Needed for VRADA experiments
EPS = torch.finfo(torch.float).eps # numerical logs

class VRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        #encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #prior
        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

        #decoder
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        self.dec_mean = nn.Linear(h_dim, x_dim)
        #self.dec_mean = nn.Sequential(
        #    nn.Linear(h_dim, x_dim),
        #    nn.Sigmoid())

        #recurrence
        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)


    def forward(self, x):

        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        nll_loss = 0

        h = torch.zeros(self.n_layers, x.size(1), self.h_dim, device=x.device)
        for t in range(x.size(0)):

            phi_x_t = self.phi_x(x[t])

            #encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_std_t = self.enc_std(enc_t) 

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            dec_std_t = self.dec_std(dec_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            #computing losses
            kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            nll_loss += self._nll_gauss(dec_mean_t, dec_std_t, x[t])
            #nll_loss += self._nll_bernoulli(dec_mean_t, x[t])

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_dec_mean.append(dec_mean_t)
            all_dec_std.append(dec_std_t)

        return kld_loss, nll_loss, z_t #\
            #(all_enc_mean, all_enc_std), \
            #(all_dec_mean, all_dec_std)


    def sample(self, seq_len):

        sample = torch.zeros(seq_len, self.x_dim, device=device)

        h = torch.zeros(self.n_layers, 1, self.h_dim, device=device)
        for t in range(seq_len):

            #prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
            phi_z_t = self.phi_z(z_t)

            #decoder
            dec_t = self.dec(torch.cat([phi_z_t, h[-1]], 1))
            dec_mean_t = self.dec_mean(dec_t)
            #dec_std_t = self.dec_std(dec_t)

            phi_x_t = self.phi_x(dec_mean_t)

            #recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)

            sample[t] = dec_mean_t.data

        return sample


    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)


    def _init_weights(self, stdv):
        pass


    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=std.device, dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)


    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""

        kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) + 
            (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
            std_2.pow(2) - 1)
        return  0.5 * torch.sum(kld_element)


    def _nll_bernoulli(self, theta, x):
        return - torch.sum(x*torch.log(theta + EPS) + (1-x)*torch.log(1-theta-EPS))


    def _nll_gauss(self, mean, std, x):
        return torch.sum(torch.log(std + EPS) + torch.log(2*torch.Tensor([torch.pi]).to(device = std.device))/2 + (x - mean).pow(2)/(2*std.pow(2)))


##################################################
##########  OTHER NETWORKS  ######################
##################################################

class codats_classifier(nn.Module):
    def __init__(self, configs):
        super(codats_classifier, self).__init__()
        model_output_dim = configs.features_len
        self.hidden_dim = configs.hidden_dim
        self.logits = nn.Sequential(
            nn.Linear(model_output_dim * configs.final_out_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, configs.num_classes))

    def forward(self, x_in):
        predictions = self.logits(x_in)
        return predictions


class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by DANN ##############
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


#### Codes required by CDAN ##############
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_CDAN(nn.Module):
    """Discriminator model for CDAN ."""

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_CDAN, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels * configs.num_classes, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


#### Codes required by AdvSKM ##############
class Cosine_act(nn.Module):
    def __init__(self):
        super(Cosine_act, self).__init__()

    def forward(self, input):
        return torch.cos(input)


cos_act = Cosine_act()

class AdvSKM_Disc(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dim, hidden_dim):
        """Init discriminator."""
        super(AdvSKM_Disc, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hidden_dim
        self.branch_1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            cos_act,
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.Linear(self.hid_dim // 2, self.hid_dim // 2),
            nn.BatchNorm1d(self.hid_dim // 2),
            cos_act
        )
        self.branch_2 = nn.Sequential(
            nn.Linear(self.input_dim, self.hid_dim),
            nn.Linear(self.hid_dim, self.hid_dim),
            nn.BatchNorm1d(self.hid_dim),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.hid_dim // 2),
            nn.Linear(self.hid_dim // 2, self.hid_dim // 2),
            nn.BatchNorm1d(self.hid_dim // 2),
            nn.ReLU())

    def forward(self, input):
        """Forward the discriminator."""
        out_cos = self.branch_1(input)
        out_rel = self.branch_2(input)
        total_out = torch.cat((out_cos, out_rel), dim=1)
        return total_out
