import argparse
import os
import datetime
import time
import random
import numpy as np 
import logging
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf 
from model.zskb import ZSKB 
from utils.data_helper import DataSet
from utils.triplet_classify import run_triplet_classify
from utils.link_prediction import run_link_prediction

logger = logging.getLogger()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_training(args):
    # ----------------------------------------------------- #

    # gpu setting
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', args.gpu_device)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)

    session_conf = tf.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=False)
    sess = tf.Session(config=session_conf)

    # Checkpoint directory
    checkpoint_dir = args.save_dir 
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    assert os.path.exists(checkpoint_dir)

    # log file
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(checkpoint_dir+'test.log', 'w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info('args: {}'.format(args))

    # ----------------------------------------------------- #

    # prepare data
    logger.info("Loading data...")
    dataset = DataSet(args, logger)
    logger.info("Loading finish...")

    # args.aggregate_type = 'test_attention'
    model = ZSKB(args, dataset.num_training_entity, dataset.num_relation)

    # saver for checkpoint and initialization
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    assert checkpoint_file != None
    logger.info('Restore the model from {}'.format(checkpoint_file))
    saver.restore(sess, checkpoint_file)

    html_file = open('visualization_mean_tail.html', 'w')
    cnt_batch = 0
    for batch_eval in dataset.batch_iter_epoch(dataset.triplets_test, 1024, corrupt=False, shuffle=True):
        batch_weight_ph, batch_weight_pt, batch_triplet, batch_relation_tail, batch_neighbor_head, batch_neighbor_tail = batch_eval
        batch_relation_head = batch_triplet[:, 1]
        # batch_weight_head, batch_weight_tail, batch_attention_head, batch_attention_tail = sess.run(
            # [model.weight_loss_ph, model.weight_loss_pt, model.attention_logit_ph, model.attention_logit_pt],
            # feed_dict={
            #         model.neighbor_head_pos: batch_neighbor_head,
            #         model.neighbor_tail_pos: batch_neighbor_tail,
            #         model.input_relation_ph: batch_relation_head,
            #         model.input_relation_pt: batch_relation_tail,
            #         model.neighbor_weight_ph: batch_weight_ph,
            #         model.neighbor_weight_pt: batch_weight_pt,
            #         model.dropout_keep_prob: 1.0
            #     })
        for id_triplet in xrange(len(batch_relation_head)):
            try:
                head = dataset.i2e[batch_triplet[id_triplet][0]]
            except:
                head = 'None'
            try:
                tail = dataset.i2e[batch_triplet[id_triplet][2]]
            except:
                tail = 'None'
            query_relation = dataset.i2r[batch_relation_head[id_triplet]]
            html_file.write('<p>' + head + ' -> ' + query_relation + ' -> ' + tail + '</p>\n')
            # html_file.write('<p>head</p>\n')

            # weight_head = batch_weight_head[id_triplet]
            # prior_weight = batch_weight_ph[id_triplet]
            # attention_weight = batch_attention_head[id_triplet]
            # # print weight_head
            # weight_head = weight_head / weight_head.max()
            # # prior_weight = softmax(np.log(prior_weight + 1e-9))
            # # prior_weight = prior_weight / prior_weight.max()
            # attention_weight = softmax(attention_weight)
            # attention_weight = attention_weight / attention_weight.max()
            # rank_weight = (-weight_head).argsort()
            # neighbors = batch_neighbor_head[id_triplet]
            # for rank in rank_weight:
            #     neighbor_relation = dataset.i2r[neighbors[rank][0]]
            #     if neighbor_relation == 'PAD':
            #         continue
            #     try:
            #         neighbor_entity = dataset.i2e[neighbors[rank][1]]
            #     except:
            #         neighbor_entity = 'None'
            #     neighbor_weight = weight_head[rank]
            #     prior_neighbor_weight = prior_weight[rank]
            #     attention_neighbor_weight = attention_weight[rank]
            #     html_file.write('<font style="background: rgba(0, 255, 255, %f)">%s</font> -> </font><font style="background: rgba(255, 255, 0, %f)">%s</font><br>\n' %\
            #          (neighbor_weight, neighbor_relation, attention_neighbor_weight, neighbor_entity))

            # html_file.write('<p>tail</p>\n')
            # weight_tail = batch_weight_tail[id_triplet]
            # prior_weight = batch_weight_pt[id_triplet]
            # attention_weight = batch_attention_tail[id_triplet]
            # weight_tail = weight_tail / weight_tail.max()
            # attention_weight = softmax(attention_weight)
            # attention_weight = attention_weight / attention_weight.max()
            # rank_weight = (-weight_tail).argsort()
            # neighbors = batch_neighbor_tail[id_triplet]
            # for rank in rank_weight:
            #     neighbor_relation = dataset.i2r[neighbors[rank][0]]
            #     if neighbor_relation == 'PAD':
            #         continue
            #     try:
            #         neighbor_entity = dataset.i2e[neighbors[rank][1]]
            #     except:
            #         neighbor_entity = 'None'
            #     neighbor_weight = weight_tail[rank]
            #     attention_neighbor_weight = attention_weight[rank]
            #     html_file.write('<font style="background: rgba(0, 255, 255, %f)">%s</font> -> </font><font style="background: rgba(255, 255, 0, %f)">%s</font><br>\n' %\
            #          (neighbor_weight, neighbor_relation, attention_neighbor_weight, neighbor_entity))

            sample_predict_head = dataset.next_sample_eval(batch_triplet[id_triplet], is_test=True)
            # rank list of head and tail prediction
            def eval_by_batch(data_eval):
                prediction_all = []
                for batch_eval in dataset.batch_iter_epoch(data_eval, 4096, corrupt=False, shuffle=False):
                    batch_weight_ph, batch_weight_pt, batch_triplet, batch_relation_tail, batch_neighbor_head, batch_neighbor_tail = batch_eval
                    batch_relation_head = batch_triplet[:, 1]
                    prediction_batch = sess.run(model.positive_score, 
                        feed_dict={
                                model.neighbor_head_pos: batch_neighbor_head,
                                model.neighbor_tail_pos: batch_neighbor_tail,
                                model.input_relation_ph: batch_relation_head,
                                model.input_relation_pt: batch_relation_tail,
                                model.neighbor_weight_ph: batch_weight_ph,
                                model.neighbor_weight_pt: batch_weight_pt,
                                model.dropout_keep_prob: 1.0
                            })
                    prediction_all.extend(prediction_batch)
                return np.asarray(prediction_all)

            prediction_head = eval_by_batch(sample_predict_head)
            sorted_rank = (-prediction_head).argsort()
            rank_head_current = sorted_rank.argmin()
            for id_rank in sorted_rank[:20]:
                prediction = sample_predict_head[id_rank][0]
                try:
                    prediction_name = dataset.i2e[prediction]
                except:
                    prediction_name = None
                if id_rank == 0:
                    html_file.write('-*-%s-*-<br>\n' % prediction_name)
                else:
                    html_file.write('%s<br>\n' % prediction_name)
            html_file.write('<p>' + '-' * 100 + '</p>')

        cnt_batch += 1
        if cnt_batch > 10:
            break
    html_file.close()

    # ----------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description='Run testing zero-shot KB model.')
    parser.add_argument('--data_dir', '-D', type=str)
    parser.add_argument('--save_dir', '-S', type=str)

    parser.add_argument('--pretrain', action='store_true')

    # model
    parser.add_argument('--use_relation', type=int, default=0)
    parser.add_argument('--embedding_dim', '-e', type=int, default=50)
    parser.add_argument('--max_neighbor', type=int, default=64)
    parser.add_argument('--n_neg', '-n', type=int, default=1)
    parser.add_argument('--aggregate_type', type=str, default='gnn_mean')
    parser.add_argument('--iter_routing', '-r', type=int, default=1)
    parser.add_argument('--score_function', type=str, default='TransE')
    parser.add_argument('--loss_function', type=str, default='margin')
    parser.add_argument('--margin', type=float, default='1.0')
    parser.add_argument('--corrupt_mode', type=str, default='both')
  
    # training
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0)
    parser.add_argument('--dis_weight', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--evaluate_size', type=int, default=1000)
    parser.add_argument('--steps_per_display', type=int, default=100)
    parser.add_argument('--epoch_per_checkpoint', type=int, default=50)

    # gpu option
    parser.add_argument('--gpu_fraction', type=float, default=0.2)
    parser.add_argument('--gpu_device', type=str, default='0')
    parser.add_argument('--allow_soft_placement', type=bool, default=False)

    args = parser.parse_args()

    run_training(args=args)

if __name__ == '__main__':
    main()