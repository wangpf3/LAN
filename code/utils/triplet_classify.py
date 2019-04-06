# coding=utf-8

import argparse
import os
import time
import datetime
import random
import tensorflow as tf 
import numpy as np

def run_triplet_classify(args, sess, model, dataset, epoch, logger, is_test=True):

    def get_score(data_eval):
        score_all = [] 
        for batch_eval in dataset.batch_iter_epoch(data_eval, args.batch_size, corrupt=False, shuffle=False):
            # batch_weight_ph, batch_weight_pt, batch_triplet, batch_relation_tail, batch_neighbor_head, batch_neighbor_tail = batch_eval
            batch_triplet, batch_relation_tail, batch_neighbor_head, batch_neighbor_tail = batch_eval
            batch_relation_head = batch_triplet[:, 1]
            score_batch = sess.run(model.positive_score, 
                feed_dict={
                        model.neighbor_head_pos: batch_neighbor_head,
                        model.neighbor_tail_pos: batch_neighbor_tail,
                        model.input_relation_ph: batch_relation_head,
                        model.input_relation_pt: batch_relation_tail,
                        # model.neighbor_weight_ph: batch_weight_ph,
                        # model.neighbor_weight_pt: batch_weight_pt,
                        model.dropout_keep_prob: 1.0
                    })
            score_all.extend(score_batch)
        score_all = np.asarray(score_all)
        return score_all 

    def get_threshold():
        score_all = get_score(dataset.triplets_dev)

        min_score = min(score_all)
        max_score = max(score_all)

        best_thresholds = []
        best_accuracies = []
        for i in xrange(dataset.num_relation):
            best_thresholds.append(min_score)
            best_accuracies.append(-1)

        score = min_score 
        increment = 0.01
        while(score <= max_score):
            for i in xrange(dataset.num_relation):
                current_relation_list = (dataset.triplets_dev[:, 1] == i)
                predictions = (score_all[current_relation_list] >= score) * 2 -1
                accuracy = np.mean(predictions == dataset.triplets_dev[current_relation_list, 3])

                if accuracy > best_accuracies[i]:
                    best_accuracies[i] = accuracy
                    best_thresholds[i] = score

            score += increment
        # logger.info('thresholds on valid set:')
        # for i, th in enumerate(best_thresholds):
        #     logger.info('{}\t{:.3f}'.format(i, th))

        return best_thresholds

    def get_prediction(triplets_test):
        score_all = get_score(triplets_test)
        best_thresholds = get_threshold()
        prediction_all = []
        for i in xrange(len(triplets_test)):
            rel = triplets_test[i, 1]
            if score_all[i] >= best_thresholds[rel]:
                prediction_all.append(1)
            else:
                prediction_all.append(-1)

        return np.asarray(prediction_all)

    if is_test:
        triplets_test = dataset.triplets_test
    else:
        triplets_test = dataset.triplets_dev
    prediction_all = get_prediction(triplets_test)
    precision = sum([1 for res in (prediction_all == triplets_test[:, 3]) if res])
    precision = precision * 100.0 / len(triplets_test) 
    logger.info('Epoch {} Triplet classify precision: {:.3f}%'.format(epoch, precision))
    return precision 

