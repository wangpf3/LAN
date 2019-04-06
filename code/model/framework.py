import tensorflow as tf
import numpy as np

class FrameWork(object):
    def __init__(self, args, num_entity, num_relation):
        self.max_neighbor = args.max_neighbor
        self.embedding_dim = args.embedding_dim
        self.learning_rate = args.learning_rate
        self.aggregate_type = args.aggregate_type
        self.score_function = args.score_function
        self.loss_function = args.loss_function
        self.use_relation = args.use_relation
        self.margin = args.margin
        self.weight_decay = args.weight_decay

        self.num_entity = num_entity
        self.num_relation = num_relation

        with tf.variable_scope('input'):

            self.neighbor_weight_ph = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.max_neighbor, 2],
                    name='neighbor_weight_ph')

            self.neighbor_weight_pt = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.max_neighbor, 2],
                    name='neighbor_weight_pt')

            self.neighbor_weight_nh = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.max_neighbor, 2],
                    name='neighbor_weight_nh')

            self.neighbor_weight_nt = tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, self.max_neighbor, 2],
                    name='neighbor_weight_nt')

            self.neighbor_head_pos = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, self.max_neighbor, 2],
                    name='neighbor_head_pos')

            self.neighbor_head_neg = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, self.max_neighbor, 2],
                    name='neighbor_head_neg')

            self.neighbor_tail_pos = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, self.max_neighbor, 2],
                    name='neighbor_tail_pos')

            self.neighbor_tail_neg = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, self.max_neighbor, 2],
                    name='neighbor_tail_neg')

            self.input_relation_ph = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None],
                    name='relation_head_pos')

            self.input_relation_pt = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None],
                    name='relation_tail_pos')

            self.input_relation_nh = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None],
                    name='relation_head_neg')

            self.input_relation_nt = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None],
                    name='relation_tail_neg')

            self.input_triplet_pos = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, 3],
                    name='input_triplet_pos')

            self.input_triplet_neg = tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, 3],
                    name='input_triplet_neg')

            self.embedding_placeholder = tf.placeholder(
                    dtype=tf.float32, 
                    shape=[self.num_entity + 1, self.embedding_dim],
                    name='embedding_placeholder'
                )


        with tf.variable_scope('embeddings'):
            self.entity_embedding = tf.get_variable(
                    name='entity_embedding',
                    shape=[self.num_entity + 1, self.embedding_dim],
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False)
            )

            self.relation_embedding_out = tf.get_variable(
                    name='relation_embedding_out',
                    shape=[self.num_relation, self.embedding_dim],
                    initializer=tf.contrib.layers.xavier_initializer(uniform=False)
            )

        # self.entity_embedding_init = self.entity_embedding.assign(self.embedding_placeholder)

        # get head, tail, relation embedded
        encoder = None
        if self.aggregate_type == 'gnn_mean':
            from aggregator import GNN_MEAN as Encoder 
            encoder = Encoder(self.num_relation, self.embedding_dim)
        elif self.aggregate_type == 'lstm':
            from aggregator import LSTM as Encoder
            encoder = Encoder(self.num_relation, self.embedding_dim)
        elif self.aggregate_type == 'attention':
            from aggregator import ATTENTION as Encoder
            encoder = Encoder(self.num_relation, self.num_entity, self.embedding_dim)
        else:
            print 'Not emplemented yet!'
        assert encoder != None

        # aggregate on neighbors input
        head_pos_embedded, self.weight_ph = self.aggregate(encoder, self.neighbor_head_pos, self.input_relation_ph, self.neighbor_weight_ph)
        tail_pos_embedded, self.weight_pt = self.aggregate(encoder, self.neighbor_tail_pos, self.input_relation_pt, self.neighbor_weight_pt)

        head_neg_embedded, _ = self.aggregate(encoder, self.neighbor_head_neg, self.input_relation_nh, self.neighbor_weight_nh)
        tail_neg_embedded, _ = self.aggregate(encoder, self.neighbor_tail_neg, self.input_relation_nt, self.neighbor_weight_nt)

        # get score
        decoder = None
        if self.score_function == 'TransE':
            from score_function import TransE as Decoder 
            decoder = Decoder()
        elif self.score_function == 'Distmult':
            from score_function import Distmult as Decoder
            decoder = Decoder()
        elif self.score_function == 'Complex':
            from score_function import Complex as Decoder
            decoder = Decoder(self.embedding_dim)
        elif self.score_function == 'Analogy':
            from score_function import Analogy as Decoder
            decoder = Decoder(self.embedding_dim)
        else:
            print 'Not emplemented yet!'
        assert decoder != None

        emb_relation_pos_out = tf.nn.embedding_lookup(self.relation_embedding_out, self.input_relation_ph)
        emb_relation_neg_out = tf.nn.embedding_lookup(self.relation_embedding_out, self.input_relation_nh)

        self.positive_score = self.score_triplet(decoder, head_pos_embedded, tail_pos_embedded, emb_relation_pos_out)
        negative_score = self.score_triplet(decoder, head_neg_embedded, tail_neg_embedded, emb_relation_neg_out)

        ph_origin_embedded = tf.nn.embedding_lookup(self.entity_embedding, self.input_triplet_pos[:, 0])
        pt_origin_embedded = tf.nn.embedding_lookup(self.entity_embedding, self.input_triplet_pos[:, 2])
        nh_origin_embedded = tf.nn.embedding_lookup(self.entity_embedding, self.input_triplet_neg[:, 0])
        nt_origin_embedded = tf.nn.embedding_lookup(self.entity_embedding, self.input_triplet_neg[:, 2])

        origin_positive_score = self.score_triplet(decoder, ph_origin_embedded, pt_origin_embedded, emb_relation_pos_out)
        origin_negative_score = self.score_triplet(decoder, nh_origin_embedded, nt_origin_embedded, emb_relation_neg_out)

        loss = None
        if self.loss_function == 'margin':
            loss = tf.reduce_mean(tf.nn.relu(self.margin - self.positive_score + negative_score))
            loss += tf.reduce_mean(tf.nn.relu(self.margin - origin_positive_score + origin_negative_score))
            loss += self.weight_decay * encoder.l2_regularization
        elif self.loss_function == 'bce':
            labels_positive = tf.ones_like(self.positive_score)
            labels_negative = tf.zeros_like(negative_score)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels_positive,
                        logits=self.positive_score))
            loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels_negative,
                        logits=negative_score))
            loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels_positive,
                        logits=origin_positive_score))
            loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=labels_negative,
                        logits=origin_negative_score))
            loss += self.weight_decay * encoder.l2_regularization
        else:
            print 'Not such loss!'
        assert loss != None

        self.loss = loss

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)

    def aggregate(self, encoder, neighbor_ids, query_relation, weight):
        neighbor_embedded = tf.nn.embedding_lookup(self.entity_embedding, neighbor_ids[:, :, 1])
        if self.use_relation == 1:
            return encoder(neighbor_embedded, neighbor_ids, query_relation, weight)
        else:
            return encoder(neighbor_embedded, neighbor_ids[:, :, 0])

    def score_triplet(self, decoder, head_embedded, tail_embedded, relation_embedded):
        score = decoder(head_embedded, tail_embedded, relation_embedded)
        return score
