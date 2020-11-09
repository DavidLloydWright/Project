import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
from replay import *

from agents.agent import Agent
from agents.memory.memory import Memory
from agents.utils import soft_update_target_network, hard_update_target_network
from agents.utils.noise import OrnsteinUhlenbeckActionNoise
import wandb

# need to refine neural modul
class QActor(nn.Module): # Interested in this

    def __init__(self, state_size1, action_size, action_parameter_size, hidden_layers=(100,), action_input_layer=0,
                 output_layer_init_std=None, activation="relu", **kwargs): # action_size number of templates, action_parameter_size number of possible words?
        super(QActor, self).__init__()
        # #print("In QACTOR", state_size1, action_size, flush = True)
        self.state_size1 = state_size1
        # self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        # self.activation = activation

        # # create layers
        # self.layers = nn.ModuleList()
        inputSize = self.state_size1 + self.action_parameter_size
        # lastHiddenLayerSize = inputSize
        # if hidden_layers is not None:
        #     nh = len(hidden_layers)
        #     self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
        #     for i in range(1, nh):
        #         self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        #     lastHiddenLayerSize = hidden_layers[nh - 1]
        # self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_size))

        # # initialise layer weights
        # for i in range(0, len(self.layers) - 1):
        #     nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
        #     nn.init.zeros_(self.layers[i].bias)
        # if output_layer_init_std is not None:
        #     nn.init.normal_(self.layers[-1].weight, mean=0., std=output_layer_init_std)
        # # else:
        # #     nn.init.zeros_(self.layers[-1].weight)
        # nn.init.zeros_(self.layers[-1].bias)
        self.t_scorer = nn.Linear(inputSize, action_size)

    def forward(self, state, action_parameters):
        # implement forward
        # negative_slope = 0.01

        x = torch.cat((state, action_parameters), dim=1)
        # num_layers = len(self.layers)
        # for i in range(0, num_layers - 1):
        #     if self.activation == "relu":
        #         x = F.relu(self.layers[i](x))
        #     elif self.activation == "leaky_relu":
        #         x = F.leaky_relu(self.layers[i](x), negative_slope)
        #     else:
        #         raise ValueError("Unknown activation function "+str(self.activation))
        # Q = self.layers[-1](x)
        Q = self.t_scorer(x)
        return Q


class ParamActor(nn.Module): # interested in this

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers, squashing_function=False,
                 output_layer_init_std=None, init_type="kaiming", activation="relu", init_std=None):
        super(ParamActor, self).__init__()

        # self.state_size = state_size
        # self.action_size = action_size
        # self.action_parameter_size = action_parameter_size
        # self.squashing_function = squashing_function
        # self.activation = activation
        # if init_type == "normal":
        #     assert init_std is not None and init_std > 0
        # assert self.squashing_function is False  # unsupported, cannot get scaling right yet

        # # create layers
        # self.layers = nn.ModuleList()
        # inputSize = self.state_size
        # lastHiddenLayerSize = inputSize
        # if hidden_layers is not None:
        #     nh = len(hidden_layers)
        #     self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
        #     for i in range(1, nh):
        #         self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
        #     lastHiddenLayerSize = hidden_layers[nh - 1]
        # self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)
        # self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        # # initialise layer weights
        # for i in range(0, len(self.layers)):
        #     if init_type == "kaiming":
        #         nn.init.kaiming_normal_(self.layers[i].weight, nonlinearity=activation)
        #     elif init_type == "normal":
        #         nn.init.normal_(self.layers[i].weight, std=init_std)
        #     else:
        #         raise ValueError("Unknown init_type "+str(init_type))
        #     nn.init.zeros_(self.layers[i].bias)
        # if output_layer_init_std is not None:
        #     nn.init.normal_(self.action_parameters_output_layer.weight, std=output_layer_init_std)
        # else:
        #     nn.init.zeros_(self.action_parameters_output_layer.weight)
        # nn.init.zeros_(self.action_parameters_output_layer.bias)

        # nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        # nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # # fix passthrough layer to avoid instability, rest of network can compensate
        # self.action_parameters_passthrough_layer.requires_grad = False
        # self.action_parameters_passthrough_layer.weight.requires_grad = False
        # self.action_parameters_passthrough_layer.bias.requires_grad = False
        self.o1_scorer = nn.Linear(128, action_parameter_size)

    def forward(self, state):
        x = state
        # negative_slope = 0.01
        # num_hidden_layers = len(self.layers)
        # for i in range(0, num_hidden_layers):
        #     if self.activation == "relu":
        #         x = F.relu(self.layers[i](x))
        #     elif self.activation == "leaky_relu":
        #         x = F.leaky_relu(self.layers[i](x), negative_slope)
        #     else:
        #         raise ValueError("Unknown activation function "+str(self.activation))
        # action_params = self.action_parameters_output_layer(x)
        # action_params += self.action_parameters_passthrough_layer(state)

        # if self.squashing_function:
        #     assert False  # scaling not implemented yet
        #     action_params = action_params.tanh()
        #     action_params = action_params * self.action_param_lim
        # # action_params = action_params / torch.norm(action_params) ## REMOVE --- normalisation layer?? for pointmass
        return self.o1_scorer(x)
        

class TDQN(nn.Module):
    def __init__(self, args, action_size,action_parameter_size, template_size, vocab_size, vocab_size_act):
        super(TDQN, self).__init__()

        self.embeddings = nn.Embedding(vocab_size_act, args.embedding_size)

        self.state_network = StateNetwork(args, vocab_size)
        
        #self.t_scorer = nn.Linear(args.hidden_size, template_size)
        
        self.args = args
        self.template_size = template_size
        self.state_size1 = 128
        self.vocab_size_act = vocab_size_act
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        #print("IN TDQN", self.state_size1, self.action_size, flush=True)
        self.t_scorer = QActor(self.state_size1, self.action_size, 2*self.action_parameter_size)
        hidden_layers = args.hidden_size
        self.o1_scorer = ParamActor(self.state_size1, self.action_size, self.action_parameter_size, None)#self.action_size, self.action_parameter_size, hidden_layers)
        self.o2_scorer = ParamActor(self.state_size1, self.action_size, self.action_parameter_size, None)
        
    def forward(self, state):
        # print("HI", flush = True)
        # print(state, flush = True)
        x, h = self.state_network(state) # x is some pre-calculated varient of the state
        # print(x, flush = True)
        q_o1 = self.o1_scorer.forward(x)
        q_o2 = self.o2_scorer.forward(x)
        o1_select = F.softmax(q_o1, dim=1)
        o2_select = F.softmax(q_o1, dim=1) # Already on cuda
        action_parameters = torch.cat((o1_select, o2_select), dim=1)
        q_t = self.t_scorer.forward(x, action_parameters)
        #q_t = QActor(nn.Module).foward(state)
         # 3 linear neural networks one for each

        return q_t, q_o1, q_o2

    def act(self, state, epsilon):
        with torch.no_grad():
            state = torch.LongTensor(state).unsqueeze(0).permute(1, 0, 2).cuda()
            q_t, q_o1, q_o2 = self.forward(state)
            t, o1, o2 = F.softmax(q_t, dim=1).multinomial(num_samples=1).item(),\
                        F.softmax(q_o1, dim=1).multinomial(num_samples=1).item(),\
                        F.softmax(q_o2, dim=1).multinomial(num_samples=1).item()
            q_t = q_t[0,t].item()
            q_o1 = q_o1[0,o1].item()
            q_o2 = q_o2[0,o2].item()
            
            print("T", t,"\n o1", o1,"\n o2", o2,"\n q_t", q_t,"\n q_o1", q_o1,"\n q_o2", q_o2)
        return t, o1, o2, q_t, q_o1, q_o2

    
    def poly_act(self, state, n_samples=512, replacement=True): # returns template and actions along with ?
        ''' Samples many times from the model, optionally with replacement. '''
        with torch.no_grad():
            state = torch.LongTensor(state).unsqueeze(0).permute(1, 0, 2).cuda()
            q_t, q_o1, q_o2 = self.forward(state)
            #print(Net()) # checking that Neural network is setup properly
            t, o1, o2 = F.softmax(q_t, dim=1).multinomial(n_samples, replacement)[0],\
                        F.softmax(q_o1, dim=1).multinomial(n_samples, replacement)[0],\
                        F.softmax(q_o2, dim=1).multinomial(n_samples, replacement)[0] # generate samples based off of score as probasilistic distribution.
            qv_t = torch.index_select(q_t, 1, t).squeeze().cpu().detach().numpy()
            qv_o1 = torch.index_select(q_o1, 1, o1).squeeze().cpu().detach().numpy()
            qv_o2 = torch.index_select(q_o2, 1, o2).squeeze().cpu().detach().numpy()
            o1_select = F.softmax(q_o1, dim=1)
            o2_select = F.softmax(q_o1, dim=1) # Already on cuda
            #o1_select[o1] = 1
            #print("O1", o1.sort(),"/nSelect", o1_select,"/nQv", qv_o1,"/nQo1", q_o1, len(q_o1[0]), len(o1), flush = True)
            action_parameters = torch.cat((o1_select, o2_select), dim=1)
            #q_t = self.template_foward(state, action_parameters)
           
            
            return t.cpu().numpy(), o1.cpu().numpy(), o2.cpu().numpy(), qv_t, qv_o1, qv_o2

    def flatten_parameters(self):
        self.state_network.flatten_parameters()



class StateNetwork(nn.Module):
    def __init__(self, args, vocab_size):
        super(StateNetwork, self).__init__()
        self.args = args

        self.enc_look = PackedEncoderRNN(vocab_size, args.hidden_size)
        self.enc_inv = PackedEncoderRNN(vocab_size, args.hidden_size)
        self.enc_ob = PackedEncoderRNN(vocab_size, args.hidden_size)
        self.enc_preva = PackedEncoderRNN(vocab_size, args.hidden_size)

        self.fcx = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.fch = nn.Linear(args.hidden_size * 4, args.hidden_size)

    def forward(self, obs):
        x_l, h_l = self.enc_look(obs[0, :, :], self.enc_look.initHidden(self.args.batch_size))
        x_i, h_i = self.enc_inv(obs[1, :, :], self.enc_inv.initHidden(self.args.batch_size))
        x_o, h_o = self.enc_ob(obs[2, :, :], self.enc_ob.initHidden(self.args.batch_size))
        x_p, h_p = self.enc_preva(obs[3, :, :], self.enc_preva.initHidden(self.args.batch_size))

        x = F.relu(self.fcx(torch.cat((x_l, x_i, x_o, x_p), dim=1)))
        h = F.relu(self.fch(torch.cat((h_l, h_i, h_o, h_p), dim=2)))

        return x, h

    def flatten_parameters(self):
        self.enc_look.flatten_parameters()
        self.enc_inv.flatten_parameters()
        self.enc_ob.flatten_parameters()
        self.enc_preva.flatten_parameters()


class PackedEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PackedEncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden=None):
        embedded = self.embedding(input).permute(1,0,2) # T x Batch x EmbDim
        if hidden is None:
            hidden = self.initHidden(input.size(0))

        # Pack the padded batch of sequences
        lengths = torch.tensor([torch.nonzero(n)[-1] + 1 for n in input], dtype=torch.long).cuda()

        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        output, hidden = self.gru(packed, hidden)
        # Unpack the padded sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        # Return only the last timestep of output for each sequence
        idx = (lengths-1).view(-1, 1).expand(len(lengths), output.size(2)).unsqueeze(0)
        output = output.gather(0, idx).squeeze(0)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()

    def flatten_parameters(self):
        self.gru.flatten_parameters()
