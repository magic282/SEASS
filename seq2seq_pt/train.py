from __future__ import division

import s2s
import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import logging

try:
    import ipdb
except ImportError:
    pass
from s2s.xinit import xavier_normal, xavier_uniform
import os
from PyRouge.Rouge import Rouge

import xargs

parser = argparse.ArgumentParser(description='train.py')
xargs.add_data_options(parser)
xargs.add_model_options(parser)
xargs.add_train_options(parser)

opt = parser.parse_args()

logging.basicConfig(format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log.txt'
if opt.log_home:
    log_file_name = os.path.join(opt.log_home, log_file_name)
file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)-5.5s:%(name)s] %(message)s'))
logging.root.addHandler(file_handler)
logger = logging.getLogger(__name__)

logger.info('My PID is {0}'.format(os.getpid()))
logger.info(opt)

if torch.cuda.is_available() and not opt.gpus:
    logger.info("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.seed > 0:
    torch.manual_seed(opt.seed)

if opt.gpus:
    if opt.cuda_seed > 0:
        torch.cuda.manual_seed(opt.cuda_seed)
    cuda.set_device(opt.gpus[0])

logger.info('My seed is {0}'.format(torch.initial_seed()))
logger.info('My cuda seed is {0}'.format(torch.cuda.initial_seed()))


def NMTCriterion(vocabSize):
    weight = torch.ones(vocabSize)
    weight[s2s.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if opt.gpus:
        crit.cuda()
    return crit


def loss_function(g_outputs, g_targets, generator, crit, eval=False):
    g_out_t = g_outputs.view(-1, g_outputs.size(2))
    g_prob_t = generator(g_out_t)

    g_loss = crit(g_prob_t, g_targets.view(-1))
    total_loss = g_loss
    report_loss = total_loss.data[0]
    return total_loss, report_loss, 0


def addPair(f1, f2):
    for x, y1 in zip(f1, f2):
        yield (x, y1)
    yield (None, None)


def load_dev_data(translator, src_file, tgt_file):
    dataset, raw = [], []
    srcF = open(src_file, encoding='utf-8')
    tgtF = open(tgt_file, encoding='utf-8')
    src_batch, tgt_batch = [], []
    for line, tgt in addPair(srcF, tgtF):
        if (line is not None) and (tgt is not None):
            src_tokens = line.strip().split(' ')
            src_batch += [src_tokens]
            tgt_tokens = tgt.strip().split(' ')
            tgt_batch += [tgt_tokens]

            if len(src_batch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(src_batch) == 0:
                break
        data = translator.buildData(src_batch, tgt_batch)
        dataset.append(data)
        raw.append((src_batch, tgt_batch))
        src_batch, tgt_batch = [], []
    srcF.close()
    tgtF.close()
    return (dataset, raw)


evalModelCount = 0
totalBatchCount = 0
rouge_calculator = Rouge.Rouge()


def evalModel(model, translator, evalData):
    global evalModelCount
    global rouge_calculator
    evalModelCount += 1
    ofn = 'dev.out.{0}'.format(evalModelCount)
    if opt.save_path:
        ofn = os.path.join(opt.save_path, ofn)
    predict, gold = [], []
    processed_data, raw_data = evalData
    for batch, raw_batch in zip(processed_data, raw_data):
        # (wrap(srcBatch), lengths), (wrap(tgtBatch), ), indices
        src, tgt, indices = batch[0]
        src_batch, tgt_batch = raw_batch

        #  (2) translate
        pred, predScore, attn, _ = translator.translateBatch(src, tgt)
        pred, predScore, attn = list(zip(
            *sorted(zip(pred, predScore, attn, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            n = 0
            predBatch.append(
                translator.buildTargetTokens(pred[b][n], src_batch[b], attn[b][n])
            )
        gold += [' '.join(r) for r in tgt_batch]
        predict += [' '.join(sents) for sents in predBatch]
    scores = rouge_calculator.compute_rouge(gold, predict)
    with open(ofn, 'w', encoding='utf-8') as of:
        for p in predict:
            of.write(p + '\n')
    return scores['rouge-2']['f'][0]


def trainModel(model, translator, trainData, validData, dataset, optim):
    logger.info(model)
    model.train()
    logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))

    # define criterion of each GPU
    criterion = NMTCriterion(dataset['dicts']['tgt'].size())

    start_time = time.time()

    def saveModel(metric=None):
        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(
            opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        save_model_path = 'model'
        if opt.save_path:
            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)
            save_model_path = opt.save_path + os.path.sep + save_model_path
        if metric is not None:
            torch.save(checkpoint, '{0}_devRouge_{1}_e{2}.pt'.format(save_model_path, round(metric, 4), epoch))
        else:
            torch.save(checkpoint, '{0}_e{1}.pt'.format(save_model_path, epoch))

    def trainEpoch(epoch):

        if opt.extra_shuffle and epoch > opt.curriculum:
            logger.info('Shuffling...')
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):
            global totalBatchCount
            totalBatchCount += 1
            # (wrap(srcBatch), lengths), (wrap(tgtBatch)), indices
            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1]  # exclude original indices

            model.zero_grad()
            g_outputs = model(batch)
            targets = batch[1][0][1:]  # exclude <s> from targets
            loss, res_loss, num_correct = loss_function(g_outputs, targets, model.generator, criterion)

            # update the parameters
            loss.backward()
            optim.step()

            num_words = targets.data.ne(s2s.Constants.PAD).sum()
            report_loss += res_loss
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += batch[0][-1].data.sum()
            total_loss += res_loss
            total_num_correct += num_correct
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                logger.info(
                    "Epoch %2d, %6d/%5d/%5d; acc: %6.2f; loss: %6.2f; words: %5d; ppl: %6.2f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                    (epoch, totalBatchCount, i + 1, len(trainData),
                     report_num_correct / report_tgt_words * 100,
                     report_loss,
                     report_tgt_words,
                     math.exp(report_loss / report_tgt_words),
                     report_src_words / (time.time() - start),
                     report_tgt_words / (time.time() - start),
                     time.time() - start))

                report_loss = report_tgt_words = report_src_words = report_num_correct = 0
                start = time.time()

            if validData is not None and totalBatchCount % opt.eval_per_batch == -1 % opt.eval_per_batch \
                    and totalBatchCount >= opt.start_eval_batch:
                model.eval()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                valid_bleu = evalModel(model, translator, validData)
                model.train()
                logger.warning("Set model to {0} mode".format('train' if model.decoder.dropout.training else 'eval'))
                model.decoder.attn.mask = None
                logger.info('Validation Score: %g' % (valid_bleu * 100))
                if valid_bleu >= optim.best_metric:
                    saveModel(valid_bleu)
                optim.updateLearningRate(valid_bleu, epoch)

        return total_loss / total_words, total_num_correct / total_words

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        logger.info('')
        global eeee
        eeee = epoch
        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        logger.info('Train perplexity: %g' % train_ppl)
        logger.info('Train accuracy: %g' % (train_acc * 100))
        logger.info('Saving checkpoint for epoch {0}...'.format(epoch))
        saveModel()


def main():
    import onlinePreprocess
    onlinePreprocess.seq_length = opt.max_sent_length
    onlinePreprocess.shuffle = 1 if opt.process_shuffle else 0
    from onlinePreprocess import prepare_data_online
    dataset = prepare_data_online(opt.train_src, opt.src_vocab, opt.train_tgt, opt.tgt_vocab)

    trainData = s2s.Dataset(dataset['train']['src'], dataset['train']['tgt'], opt.batch_size, opt.gpus)
    dicts = dataset['dicts']
    logger.info(' * vocabulary size. source = %d; target = %d' %
                (dicts['src'].size(), dicts['tgt'].size()))
    logger.info(' * number of training sentences. %d' %
                len(dataset['train']['src']))
    logger.info(' * maximum batch size. %d' % opt.batch_size)

    logger.info('Building model...')

    encoder = s2s.Models.Encoder(opt, dicts['src'])
    decoder = s2s.Models.Decoder(opt, dicts['tgt'])
    decIniter = s2s.Models.DecInit(opt)

    generator = nn.Sequential(
        nn.Linear(opt.dec_rnn_size // opt.maxout_pool_size, dicts['tgt'].size()),  # TODO: fix here
        nn.LogSoftmax())

    model = s2s.Models.NMTModel(encoder, decoder, decIniter)
    model.generator = generator
    translator = s2s.Translator(opt, model, dataset)

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    for pr_name, p in model.named_parameters():
        logger.info(pr_name)
        # p.data.uniform_(-opt.param_init, opt.param_init)
        if p.dim() == 1:
            # p.data.zero_()
            p.data.normal_(0, math.sqrt(6 / (1 + p.size(0))))
        else:
            xavier_normal(p, math.sqrt(3))
            # xavier_uniform(p)

    encoder.load_pretrained_vectors(opt)
    decoder.load_pretrained_vectors(opt)

    optim = s2s.Optim(
        opt.optim, opt.learning_rate,
        max_grad_norm=opt.max_grad_norm,
        max_weight_value=opt.max_weight_value,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        decay_bad_count=opt.halve_lr_bad_count
    )

    optim.set_parameters(model.parameters())

    validData = None
    if opt.dev_input_src and opt.dev_ref:
        validData = load_dev_data(translator, opt.dev_input_src, opt.dev_ref)
    trainModel(model, translator, trainData, validData, dataset, optim)


if __name__ == "__main__":
    main()
