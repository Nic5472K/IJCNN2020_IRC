###===###
# NicK of the ANU
#---
# Comments:
# (1)   The backbone RNN code for IRC-SRU
#
# (2)   Mainly built on top of SalesForce's code (SF)
#       https://github.com/salesforce/awd-lstm-lm

###===###
import  torch
import  torch.nn                as      nn
import  torch.nn.functional     as      F
#
#---
import  math
#
#---
from    SF_embed_regularize     import  embedded_dropout
from    SF_locked_dropout       import  LockedDropout
from    SF_weight_drop2         import  WeightDrop

###===###
class Method(nn.Module):
    def __init__(self,ID, HD, dropout):
        super(Method, self).__init__()

        self.ih     = nn.Linear(ID, HD * 2, bias = False)
        self.WA     = nn.Linear(ID, HD, bias = False)

        self.uV_raw = nn.Parameter(torch.randn(HD))
        
        self.bD     = nn.Parameter(torch.randn(HD))

        self.dropout  = dropout

        self.HD = HD
        self.ID = ID

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.HD)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

        self.PFB = nn.Parameter(     torch.ones(self.HD))
        self.HWb = nn.Parameter(-1 * torch.ones(self.HD))
        self.uV2 = nn.Parameter(torch.ones(self.ID) * 2)

    def forward(self, x, HC):

        raw_w2 = self.uV_raw
        w2 = None
        w2 = F.dropout(raw_w2, p=self.dropout, training=self.training)
        setattr(self, 'uV', w2)
            
        (Q_k, S_k) = HC
        Q_k = Q_k.squeeze(0)
        S_k = S_k.squeeze(0)
        tspan = x.shape[0]

        YALL = []

        for itr in range(tspan):
            
            X_k = x[itr]

            V_k = X_k + torch.sigmoid(self.uV2) * torch.tanh(self.uV) * Q_k

            Fi, Ri = self.ih(V_k).chunk(2, dim = 1)

            F_k = torch.sigmoid(Fi + self.PFB)
            R_k = torch.sigmoid(Ri + self.HWb)
            A_k = self.WA(X_k)
        
            
            S_k = F_k * S_k + (1 - F_k) * A_k
            
            Q_k = R_k * S_k + (1 - R_k) * X_k

            YALL.append(Q_k.unsqueeze(0))

        Q_k = Q_k.unsqueeze(0)
        S_k = S_k.unsqueeze(0)
        
        YALL2 = torch.cat(YALL)

        return YALL2, (Q_k, S_k)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [Method(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), dropout = wdrop) for l in range(nlayers)]
            #if wdrop:
            #    self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        print(self.rnns)
        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                    for l in range(self.nlayers)]
