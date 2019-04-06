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
from model.framework import FrameWork
from utils.data_helper import DataSet
from utils.triplet_classify import run_triplet_classify
from utils.link_prediction import run_link_prediction

logger = logging.getLogger()

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
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # log file
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(checkpoint_dir+'train.log', 'w')
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

    model = FrameWork(args, dataset.num_training_entity, dataset.num_relation)

    # saver for checkpoint and initialization
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())

    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint_file != None:
        logger.info('Restore the model from {}'.format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        logger.info('Start testing on checkpoints')
        if dataset.task == 'triplet_classify':
            run_triplet_classify(args, sess, model, dataset, 0, logger, is_test=True)
        else:
            run_link_prediction(args, sess, model, dataset, 0, logger, is_test=True)
        logger.info('Testing finish')
        return

    # ----------------------------------------------------- #

    # training
    num_batch = dataset.num_sample // args.batch_size
    logger.info('Train with {} batches'.format(num_batch))

    best_performance = 0.
    for epoch in xrange(args.num_epoch):
        st_epoch = time.time()
        loss_epoch = 0.
        cnt_batch = 0
        for batch_data in dataset.batch_iter_epoch(dataset.triplets_train, args.batch_size, args.n_neg):
            st_batch = time.time()
            batch_weight_ph, batch_weight_pt, batch_weight_nh, batch_weight_nt, batch_positive, batch_negative, batch_relation_ph, batch_relation_pt, batch_relation_nh, batch_relation_nt, batch_neighbor_hp, batch_neighbor_tp, batch_neighbor_hn, batch_neighbor_tn = batch_data
            # batch_positive, batch_negative, batch_relation_ph, batch_relation_pt, batch_relation_nh, batch_relation_nt, batch_neighbor_hp, batch_neighbor_tp, batch_neighbor_hn, batch_neighbor_tn = batch_data
            feed_dict = {
                    model.neighbor_head_pos: batch_neighbor_hp,
                    model.neighbor_tail_pos: batch_neighbor_tp,
                    model.neighbor_head_neg: batch_neighbor_hn,
                    model.neighbor_tail_neg: batch_neighbor_tn,
                    model.input_relation_ph: batch_relation_ph,
                    model.input_relation_pt: batch_relation_pt,
                    model.input_relation_nh: batch_relation_nh,
                    model.input_relation_nt: batch_relation_nt,
                    model.input_triplet_pos: batch_positive,
                    model.input_triplet_neg: batch_negative,
                    model.neighbor_weight_ph: batch_weight_ph,
                    model.neighbor_weight_pt: batch_weight_pt,
                    model.neighbor_weight_nh: batch_weight_nh,
                    model.neighbor_weight_nt: batch_weight_nt
                }

            _, loss_batch, _step = sess.run(
                    [model.train_op, model.loss, model.global_step], 
                    feed_dict=feed_dict
                )
            cnt_batch += 1
            loss_epoch += loss_batch
            en_batch = time.time()

            # print an overview every some batches
            if (cnt_batch+1) % args.steps_per_display == 0 or (cnt_batch+1) == num_batch:
                logger.info('epoch {}, batch {}, loss: {:.3f}, time: {:.3f}s'.format(
                    epoch, cnt_batch, loss_batch, en_batch - st_batch))

        en_epoch = time.time()
        logger.info('epoch {}, mean loss: {:.3f}, time: {:.3f}s'.format(
            epoch,
            loss_epoch / cnt_batch,
            en_epoch - st_epoch
        ))

        # evaluate the model every some steps 
        if (epoch + 1) % args.epoch_per_checkpoint == 0 or (epoch + 1) == args.num_epoch:
            st_test = time.time()
            if dataset.task == 'triplet_classify':
                performance = run_triplet_classify(args, sess, model, dataset, epoch, logger, is_test=False)
            else:
                performance = run_link_prediction(args, sess, model, dataset, epoch, logger, is_test=False)
            if performance > best_performance:
                best_performance = performance
                save_path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                time_str = datetime.datetime.now().isoformat()
                logger.info('{}: model at epoch {} save in file {}'.format(time_str, epoch, save_path))
            en_test = time.time()
            logger.info('testing finished with time: {:.3f}s'.format(en_test - st_test))

    logger.info('Training finished')
    # if dataset.task == 'triplet_classify':
    #     run_triplet_classify(args, sess, model, dataset, epoch, logger, is_test=True)
    # else:
    #     run_link_prediction(args, sess, model, dataset, epoch, logger, is_test=True)
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    logger.info('Restore the model from {}'.format(checkpoint_file))
    saver.restore(sess, checkpoint_file)
    st_test = time.time()
    if dataset.task == 'triplet_classify':
        run_triplet_classify(args, sess, model, dataset, epoch, logger, is_test=True)
    else:
        run_link_prediction(args, sess, model, dataset, epoch, logger, is_test=True)
    en_test = time.time()
    logger.info('Testing finished with time: {:.3f}s'.format(en_test - st_test))

def main():
    parser = argparse.ArgumentParser(description='Run training zero-shot KB model.')
    parser.add_argument('--data_dir', '-D', type=str)
    parser.add_argument('--save_dir', '-S', type=str)

    # model
    parser.add_argument('--use_relation', type=int, default=0)
    parser.add_argument('--embedding_dim', '-e', type=int, default=50)
    parser.add_argument('--max_neighbor', type=int, default=64)
    parser.add_argument('--n_neg', '-n', type=int, default=1)
    parser.add_argument('--aggregate_type', type=str, default='gnn_mean')
    parser.add_argument('--score_function', type=str, default='TransE')
    parser.add_argument('--loss_function', type=str, default='margin')
    parser.add_argument('--margin', type=float, default='1.0')
    parser.add_argument('--corrupt_mode', type=str, default='both')
  
    # training
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', '-w', type=float, default=0.0)
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