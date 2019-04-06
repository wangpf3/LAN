import argparse
import os
import time
import datetime
import random
import numpy as np

def run_link_prediction(args, sess, model, dataset, epoch, logger, is_test=False):
    logger.info('evaluating the current model...')
    rank_head = 0
    hit10_head = 0
    hit3_head = 0
    max_rank_head = 0
    min_rank_head = None
    acc_head = 0
    rec_rank_head = 0

    if is_test:
        evaluate_size = len(dataset.triplets_test)
        evaluate_data = dataset.triplets_test
    else:
        if args.evaluate_size == 0:
            evaluate_size = len(dataset.triplets_dev)
        else:
            evaluate_size = args.evaluate_size
        evaluate_data = dataset.triplets_dev

    cnt_sample = 0
    for triplet in random.sample(evaluate_data, evaluate_size):
        sample_predict_head = dataset.next_sample_eval(triplet, is_test=is_test)
        def eval_by_batch(data_eval):
            prediction_all = []
            for batch_eval in dataset.batch_iter_epoch(data_eval, 4096, corrupt=False, shuffle=False):
                batch_weight_ph, batch_weight_pt, batch_triplet, batch_relation_tail, batch_neighbor_head, batch_neighbor_tail = batch_eval
                # batch_triplet, batch_relation_tail, batch_neighbor_head, batch_neighbor_tail = batch_eval
                batch_relation_head = batch_triplet[:, 1]
                prediction_batch = sess.run(model.positive_score, 
                    feed_dict={
                            model.neighbor_head_pos: batch_neighbor_head,
                            model.neighbor_tail_pos: batch_neighbor_tail,
                            model.input_relation_ph: batch_relation_head,
                            model.input_relation_pt: batch_relation_tail,
                            model.neighbor_weight_ph: batch_weight_ph,
                            model.neighbor_weight_pt: batch_weight_pt,
                        })
                prediction_all.extend(prediction_batch)
            return np.asarray(prediction_all)

        prediction_head = eval_by_batch(sample_predict_head)

        rank_head_current = (-prediction_head).argsort().argmin() + 1

        rank_head += rank_head_current
        rec_rank_head += 1.0 / rank_head_current
        if rank_head_current <= 10:
          hit10_head += 1
        if rank_head_current <= 3:
          hit3_head += 1
        if max_rank_head < rank_head_current:
          max_rank_head = rank_head_current
        if min_rank_head == None:
          min_rank_head = rank_head_current
        elif min_rank_head > rank_head_current:
          min_rank_head = rank_head_current
        if rank_head_current == 1:
            acc_head += 1

    rank_head_mean = rank_head // evaluate_size 
    hit10_head = hit10_head * 1.0 / evaluate_size
    hit3_head = hit3_head * 1.0 / evaluate_size
    acc_head = acc_head * 1.0 / evaluate_size
    rec_rank_head = rec_rank_head / evaluate_size

    logger.info('epoch {} MR: {:d}, MRR: {:.3f}, hit@10: {:.3f}%, hit@3: {:.3f}%, hit@1: {:.3f}%'.format(\
          epoch, rank_head_mean, rec_rank_head, hit10_head * 100, hit3_head * 100, acc_head * 100))
    return rec_rank_head

