###===###
# NicK of the ANU
#---
# Comments:
# (1)   A IRC-GRU language model for PTB
#
# (2)   Mainly built on top of SalesForce's code (SF)
#       https://github.com/salesforce/awd-lstm-lm
#
# (3)   The PTB dataset can be found in many places
#       such as
#       https://github.com/hjc18/language_modeling_lstm/tree/master/input
#
# (4)   Original results gained from
#           PyTorch version 0.4.1.post2
#       Not PyTorch version 1.1.0

###===###
import  torch
import  torch.nn        as      nn
#
#---
import  argparse
import  time
import  math
import  numpy           as      np
#
#---
import  SF_data
import  NicK006b_model_IRCGRU  as      model

from    SF_utils        import  batchify, get_batch, repackage_hidden

###===###
parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
#
#---
# The hyperparameters that we care about in our paper
parser.add_argument('--emsize',     type=int,   default=650,
                                    help='size of word embeddings')
parser.add_argument('--nhid',       type=int,   default=650,
                                    help='number of hidden units per layer')
parser.add_argument('--nlayers',    type=int,   default=2,
                                    help='number of layers')
parser.add_argument('--epochs',     type=int,   default=50,
                                    help='upper epoch limit')
parser.add_argument('--lr',                 type=float, default=10)
#
#---
# The rest, see the SF code
parser.add_argument('--data',               type=str,   default='./penn/')
parser.add_argument('--model',              type=str,   default='LSTM')    # <--- dummy placeholder
parser.add_argument('--clip',               type=float, default=0.25)
parser.add_argument('--batch_size',         type=int,   default=20, metavar='N')
parser.add_argument('--bptt',               type=int,   default=70)
parser.add_argument('--dropout',            type=float, default=0.4)
parser.add_argument('--dropouth',           type=float, default=0.25)
parser.add_argument('--dropouti',           type=float, default=0.4)
parser.add_argument('--dropoute',           type=float, default=0.1)
parser.add_argument('--wdrop',              type=float, default=0.25)
parser.add_argument('--seed',               type=int,   default=141)
parser.add_argument('--nonmono',            type=int,   default=5)
parser.add_argument('--cuda',               action='store_false')
parser.add_argument('--log-interval',       type=int,   default=40, metavar='N')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save',               type=str,   default=randomhash+'.pt')
parser.add_argument('--alpha',              type=float, default=2)
parser.add_argument('--beta',               type=float, default=1)
parser.add_argument('--wdecay',             type=float, default=1.2e-6)
parser.add_argument('--resume',             type=str,   default='')
parser.add_argument('--optimizer',          type=str,   default='sgd')
parser.add_argument('--when', nargs="+",    type=int,   default=[-1])
args = parser.parse_args()
args.tied = True
#
#---
np.random.seed(     args.seed)
torch.manual_seed(  args.seed)

###===###
# Building the Dataset
#---
# See the SF code
def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)
def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = SF_data.Corpus(args.data)
    torch.save(corpus, fn)
eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###===###
# Building the model
#---
# See the SF code
from splitcross import SplitCrossEntropyLoss
criterion = None
ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
###
if args.resume:
    print('Resuming model ...')
    model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop
        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
if not criterion:
    splits = []
    if ntokens > 500000:
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
params = list(model.parameters()) + list(criterion.parameters())

###===###
# For training and testing
#---
# See the SF code

def evaluate(data_source, batch_size=10):
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()
        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        loss = raw_loss
        if args.alpha: loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()
        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        batch += 1
        i += seq_len

###===###
# The actual stuff starts here
#---
# See the SF code
lr = args.lr
best_val_loss = []
stored_loss = 100000000
try:
    optimizer = None
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                if 'ax' in optimizer.state[prm]:
                    prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)
            if val_loss2 < stored_loss:
                model_save(args.save)
                print('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()
        else:
            val_loss = evaluate(val_data, eval_batch_size)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            if val_loss < stored_loss:
                model_save(args.save)
                print('Saving model (new best validation)')
                stored_loss = val_loss
            if epoch == 30:
                print('Switching to ASGD')
                optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            if epoch in args.when:
                print('Saving model before learning rate decreased')
                model_save('{}.e{}'.format(args.save, epoch))
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.
            best_val_loss.append(val_loss)
        if np.mod(epoch, 5) == 0:
            torch.save(model.state_dict(), 'L3E50_EINS_prm_epoch{}'.format(epoch))
            
###===###
# Post training related
#---
# See the SF code
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')
model_load(args.save)
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
