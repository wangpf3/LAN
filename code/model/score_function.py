import tensorflow as tf
import numpy as np

class TransE(object):
    """docstring for TransE"""
    def __init__(self):
        super(TransE, self).__init__()

    def __call__(self, head, tail, relation):
        # head = tf.clip_by_norm(head, 1.0, 1)
        head = tf.nn.l2_normalize(head, 1)
        relation = tf.nn.l2_normalize(relation, 1)
        tail = tf.nn.l2_normalize(tail, 1)
        # head = tf.nn.tanh(head)
        # relation = tf.nn.tanh(relation)
        # tail = tf.nn.tanh(tail)
        dissimilarity = tf.reduce_sum(tf.abs(head + relation - tail), 1)
        score = -dissimilarity
        return score 

class Distmult(object):
    def __init__(self):
        super(Distmult, self).__init__()

    def __call__(self, head, tail, relation):
        score = tf.reduce_sum(head * relation * tail, 1)
        return score 
  
class Dotproduct(object):
    """docstring for Dotproduct"""
    def __init__(self):
        super(Dotproduct, self).__init__()

    def __call__(self, head, tail, relation):
        score =  tf.reduce_sum(head * tail, 1)
        return score
        
class TransL(object):
    def __init__(self):
        super(TransL, self).__init__()

    def __call__(self, head, tail, relation):
        head = tf.nn.l2_normalize(head, 1)
        tail = tf.nn.l2_normalize(tail, 1)
        score =  tf.reduce_sum(tf.abs(head - tail), 1)
        return -score

class Complex(object):
    """docstring for Complex"""
    def __init__(self, embedding_dimension):
        super(Complex, self).__init__()
        self.embedding_dimension = embedding_dimension
        
    def __call__(self, head, tail, relation):
        offset = self.embedding_dimension / 2
        h1 = head[:, 0:offset]
        h2 = head[:, offset:]
        r1 = relation[:, 0:offset]
        r2 = relation[:, offset:]
        t1 = tail[:, 0:offset]
        t2 = tail[:, offset:]

        score = tf.reduce_sum(h1 * t1 * r1 + h2 * t2 * r1 + h1 * t2 * r2 - h2 * t1 * r2, 1)
        return score

class Analogy(object):
    def __init__(self, embedding_dimension):
        super(Analogy, self).__init__()
        self.embedding_dimension = embedding_dimension
        
    def __call__(self, head, tail, relation):
        offset = self.embedding_dimension / 4
        score = tf.reduce_sum(relation[:, 0:offset*2]*head[:, 0:offset*2]*tail[:,0:offset*2], 1)
        h1 = head[:, offset*2:offset*3]
        h2 = head[:, offset*3:]
        r1 = relation[:, offset*2:offset*3]
        r2 = relation[:, offset*3:]
        t1 = tail[:, offset*2:offset*3]
        t2 = tail[:, offset*3:]

        score += tf.reduce_sum(h1 * t1 * r1 + h2 * t2 * r1 + h1 * t2 * r2 - h2 * t1 * r2, 1)
        return score
