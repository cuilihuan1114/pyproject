# -*- coding: utf-8 -*-
import tensorflow.compat.v1    as     tf 
tf.disable_eager_execution()  #关闭eager运算
tf.disable_v2_behavior()    #禁用TensorFlow 2.x行为
import time 
import numpy as np
# In[]
class DLOption(object):

    """Docstring for DLOption. """

    def __init__(self, epoches, learning_rate, batchsize, momentum, penaltyL2,
                 dropoutProb):
        """TODO: to be defined1.

        :epoches: TODO
        :learning_rate: TODO
        :batchsize: TODO
        :momentum: TODO
        :penaltyL2: TODO
        :dropout: TODO
        :dropoutProb: TODO

        """
        self._epoches = epoches
        self._learning_rate = learning_rate
        self._batchsize = batchsize
        self._momentum = momentum
        self._penaltyL2 = penaltyL2
        self._dropoutProb = dropoutProb

# In[]
class RBM(object):

    """RBM class for tensorflow"""

    def __init__(self, name, input_size, output_size, opts):
        """Initialize a rbm object.

        :name: TODO
        :input_size: TODO
        :output_size: TODO

        """
        
        tf.set_random_seed(0)
        np.random.seed(0)
        self._name = name
        self._input_size = input_size
        self._output_size = output_size
        self._opts = opts
        self.init_w = np.zeros([input_size, output_size], np.float32)
        # self.init_w = np.random.rand(input_size,output_size).astype(np.float32)
        self.init_hb = np.zeros([output_size], np.float32)
        # self.init_hb = np.random.rand(output_size).astype(np.float32)
        self.init_vb = np.zeros([input_size], np.float32)
        # self.init_vb = np.random.rand(input_size).astype(np.float32)
        self.w = np.zeros([input_size, output_size], np.float32)
        self.hb = np.zeros([output_size], np.float32)
        self.vb = np.zeros([input_size], np.float32)

    def reset_init_parameter(self, init_weights, init_hbias, init_vbias):
        """TODO: Docstring for reset_para.

        :init_weights: TODO
        :init_hbias: TODO
        :init_vbias: TODO
        :returns: TODO

        """
        self.init_w = init_weights
        self.init_hb = init_hbias
        self.init_vb = init_vbias

    def propup(self, visible, w, hb):
        """TODO: Docstring for propup.

        :visible: TODO
        :returns: TODO

        """
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    def propdown(self, hidden, w, vb):
        """TODO: Docstring for propdown.

        :hidden: TODO
        :returns: TODO

        """
        return tf.nn.sigmoid(
            tf.matmul(hidden, tf.transpose(w)) + vb)


    def sample_prob(self, probs):
        """TODO: Docstring for sample_prob.

        :probs: TODO
        :returns: TODO

        """

        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))

    def train(self, X):
        """TODO: Docstring for train.

        :X: TODO
        :returns: TODO

        """
#        print('pretrain rbms')
        _w = tf.placeholder("float", [self._input_size, self._output_size])
        _hb = tf.placeholder("float", [self._output_size])
        _vb = tf.placeholder("float", [self._input_size])
        _vw = tf.placeholder("float", [self._input_size, self._output_size])
        _vhb = tf.placeholder("float", [self._output_size])
        _vvb = tf.placeholder("float", [self._input_size])
        _current_vw = np.zeros(
            [self._input_size, self._output_size], np.float32)
        _current_vhb = np.zeros([self._output_size], np.float32)
        _current_vvb = np.zeros([self._input_size], np.float32)
        
        v0 = tf.placeholder("float", [None, self._input_size])
        
        h0 = self.sample_prob(self.propup(v0, _w, _hb))
        v1 = self.sample_prob(self.propdown(h0, _w, _vb))
        h1 = self.propup(v1, _w, _hb)
        positive_grad = tf.matmul(tf.transpose(v0), h0)
        negative_grad = tf.matmul(tf.transpose(v1), h1)
        update_vw = _vw * self._opts._momentum + self._opts._learning_rate *\
            (positive_grad - negative_grad) / tf.to_float(tf.shape(v0)[0])
        update_vvb = _vvb * self._opts._momentum + \
            self._opts._learning_rate * tf.reduce_mean(v0 - v1, 0)
        update_vhb = _vhb * self._opts._momentum + \
            self._opts._learning_rate * tf.reduce_mean(h0 - h1, 0)
        update_w = _w + _vw
        update_vb = _vb + _vvb
        update_hb = _hb + _vhb
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            old_w = self.init_w
            old_hb = self.init_hb
            old_vb = self.init_vb
            for i in range(self._opts._epoches):
                for start, end in zip(range(0, len(X), self._opts._batchsize),
                                      range(self._opts._batchsize,
                                            len(X), self._opts._batchsize)):
                    batch = X[start:end]
                    _current_vw = sess.run(update_vw, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vw: _current_vw})
                    _current_vhb = sess.run(update_vhb, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vhb: _current_vhb})
                    _current_vvb = sess.run(update_vvb, feed_dict={
                        v0: batch, _w: old_w, _hb: old_hb, _vb: old_vb,
                        _vvb: _current_vvb})
                    old_w = sess.run(update_w, feed_dict={
                                     _w: old_w, _vw: _current_vw})
                    old_hb = sess.run(update_hb, feed_dict={
                        _hb: old_hb, _vhb: _current_vhb})
                    old_vb = sess.run(update_vb, feed_dict={
                        _vb: old_vb, _vvb: _current_vvb})

            self.w = old_w
            self.hb = old_hb
            self.vb = old_vb

    def rbmup(self, X):
        """TODO: Docstring for rbmup.

        :X: TODO
        :returns: TODO

        """
        input_X = tf.constant(X)
        _w = tf.constant(self.w)
        _hb = tf.constant(self.hb)
        out = tf.nn.sigmoid(tf.matmul(input_X, _w) + _hb)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(out)

# In[]
class DBN(object):

    """Docstring for DBN. """

    def __init__(self, sizes, opts, X):
        """TODO: to be defined1.

        :sizes: TODO
        :opts: TODO

        """
        self._sizes = sizes
        self._opts = opts
        self._X = X
        self.rbm_list = []
        input_size = X.shape[1]
        for i, size in enumerate(self._sizes):
            self.rbm_list.append(RBM("rbm%d" % i, input_size, size, self._opts))
            input_size = size

    def train(self):
        """TODO: Docstring for train.
        :returns: TODO

        """
        X = self._X
        for rbm in self.rbm_list:
            start=time.time()
            
            rbm.train(X)
            X = rbm.rbmup(X)
            print('耗时:',time.time()-start)
# In[]
class NN(object):

    """Docstring for NN. """

    def __init__(self, sizes, opts, X, Y,X_test,Y_test,types=0):
        """TODO: to be defined1.

        :sizes: TODO
        :opts: TODO
        :X: TODO

        """
        tf.set_random_seed(0)
        np.random.seed(0)
        self._sizes = sizes
        self._opts = opts
        self._X = X
        self._Y = Y
        self._X_test = X_test
        self._Y_test = Y_test
        self.w_list = []
        self.b_list = []
        self.types=types
        input_size = X.shape[1]
        for size in self._sizes + [Y.shape[1]]:
            # max_range = 4 * np.sqrt(6. / (input_size + size))
            # self.w_list.append(
            #     np.random.uniform(
            #         -max_range, max_range, [input_size, size]
            #     ).astype(np.float32))
            self.w_list.append(
                np.zeros([input_size, size],np.float32))
            self.b_list.append(np.zeros([size], np.float32))
            input_size = size

    def load_from_dbn(self, dbn):
        """TODO: Docstring for load_from_dbn.

        :dbn: TODO
        :returns: TODO

        """
        assert len(dbn._sizes) == len(self._sizes)
        for i in range(len(self._sizes)):
            assert dbn._sizes[i] == self._sizes[i]
        for i in range(len(self._sizes)):
            self.w_list[i] = dbn.rbm_list[i].w
            self.b_list[i] = dbn.rbm_list[i].hb

    def train(self):
        
        """TODO: Docstring for train.
        :returns: TODO

        """
#        print('fine-tunning')
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * (len(self._sizes) + 1)
        _b = [None] * (len(self._sizes) + 1)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        y = tf.placeholder("float", [None, self._Y.shape[1]])
        for i in range(len(self._sizes) + 1):
            _w[i] = tf.Variable(self.w_list[i])
            _b[i] = tf.Variable(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            if i == len(self._sizes) + 1:#输出层不需要激活
                _a[i] = tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1]
            else:
                _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
        # cost=tf.nn.softmax_cross_entropy_with_logits_v2(logits=_a[-1], labels=y)
        # cost = tf.reduce_sum(cost)
        cost = tf.reduce_mean(tf.square(_a[-1] - y))
        train_op = tf.train.AdamOptimizer(
            self._opts._learning_rate, self._opts._momentum).minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            all_loss=0

            self.train_loss=[]
            self.test_loss=[]
            
            for i in range(self._opts._epoches):
                for start, end in zip(
                    range(
                        0, len(self._X),
                        self._opts._batchsize),
                    range(
                        self._opts._batchsize, len(
                            self._X),
                        self._opts._batchsize)):
                    sess.run(train_op, feed_dict={
                        _a[0]: self._X[start:end], y: self._Y[start:end]})
                    
                if self.types==1:
                    all_loss=sess.run(cost, feed_dict={_a[0]: self._X, y: self._Y})
                    self.train_loss.append(all_loss)                
                    all_loss=sess.run(cost, feed_dict={_a[0]: self._X_test, y: self._Y_test})
                    self.test_loss.append(all_loss)
                
                for i in range(len(self._sizes) + 1):
                    self.w_list[i] = sess.run(_w[i])
                    self.b_list[i] = sess.run(_b[i])
                

    def predict(self, X):
        """TODO: Docstring for predict.

        :X: TODO
        :returns: TODO

        """
        _a = [None] * (len(self._sizes) + 2)
        _w = [None] * len(self.w_list)
        _b = [None] * len(self.b_list)
        _a[0] = tf.placeholder("float", [None, self._X.shape[1]])
        for i in range(len(self.w_list)):
            _w[i] = tf.constant(self.w_list[i])
            _b[i] = tf.constant(self.b_list[i])
        for i in range(1, len(self._sizes) + 2):
            if i == len(self._sizes) + 1:#输出层不需要激活
                _a[i] = tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1]
            else:
                _a[i] = tf.nn.sigmoid(tf.matmul(_a[i - 1], _w[i - 1]) + _b[i - 1])
            
        predict_op = _a[-1]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return sess.run(predict_op, feed_dict={_a[0]: X})
