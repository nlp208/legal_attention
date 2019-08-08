import tensorflow as tf


class LegalAtt:
    def __init__(self, config, embedding_matrix, is_training):
        self.accu_num = config.accu_num
        self.art_num = config.art_num
        self.impr_num = config.impr_num

        self.top_k = config.top_k
        self.threshold = config.threshold
        self.max_seq_len = config.sequence_len

        self.kernel_size = config.kernel_size
        self.filter_dim = config.filter_dim
        self.att_size = config.att_size
        self.fc_size = config.fc_size_s

        self.embedding_matrix = tf.get_variable(
            initializer=tf.constant_initializer(embedding_matrix),
            shape=embedding_matrix.shape,
            trainable=config.embedding_trainable,
            dtype=tf.float32,
            name='embedding_matrix'
        )
        self.embedding_size = embedding_matrix.shape[-1]

        self.lr = config.lr
        self.optimizer = config.optimizer
        self.dropout = config.dropout
        self.l2_rate = config.l2_rate
        self.use_batch_norm = config.use_batch_norm

        self.is_training = is_training

        self.w_init = tf.truncated_normal_initializer(stddev=0.1)
        self.b_init = tf.constant_initializer(0.1)

        if self.l2_rate > 0.0:
            self.regularizer = tf.keras.regularizers.l2(self.l2_rate)
        else:
            self.regularizer = None

        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.fact = tf.placeholder(dtype=tf.int32, shape=[None, self.max_seq_len], name='fact')
        self.fact_len = tf.placeholder(dtype=tf.int32, shape=[None], name='fact_len')
        self.art = tf.placeholder(dtype=tf.int32, shape=[None, self.art_num, self.max_seq_len], name='art')
        self.art_len = tf.placeholder(dtype=tf.int32, shape=[None, self.art_num], name='art_len')
        self.accu = tf.placeholder(dtype=tf.float32, shape=[None, self.accu_num], name='accu')
        self.relevant_art = tf.placeholder(dtype=tf.float32, shape=[None, self.art_num], name='relevant_art')
        self.impr = tf.placeholder(dtype=tf.float32, shape=[None, self.impr_num], name='impr')

        with tf.variable_scope('fact_embedding'):
            fact_em = self.embedding_layer(self.fact)

        with tf.variable_scope('fact_encoder'):
            fact_enc = self.cnn_encoder(fact_em)

        with tf.variable_scope('article_extractor'):
            art_score, top_k_score, top_k_indices = self.get_top_k_indices(fact_enc)
            top_k_art, top_k_art_len = self.get_top_k_articles(top_k_indices)

        with tf.variable_scope('article_embedding'):
            top_k_art_em = self.embedding_layer(top_k_art)

        with tf.variable_scope('article_encoder'):
            shared_layers = {}
            for kernel_size in self.kernel_size:
                shared_layers['conv_' + str(kernel_size)] = tf.keras.layers.Conv1D(
                    self.filter_dim,
                    kernel_size,
                    padding='same',
                    kernel_regularizer=self.regularizer,
                    name='conv_' + str(kernel_size)
                )
                if self.use_batch_norm:
                    shared_layers['norm_' + str(kernel_size)] = tf.keras.layers.BatchNormalization(name='norm_' + str(kernel_size))

            top_k_art_enc = self.art_encoder(top_k_art_em, shared_layers)

        with tf.variable_scope('attention'):
            key = tf.keras.layers.Dense(
                self.att_size,
                tf.nn.tanh,
                use_bias=False,
                kernel_regularizer=self.regularizer
            )(fact_enc)

            ones = tf.ones_like(top_k_score, dtype=tf.float32)
            zeros = tf.zeros_like(top_k_score, dtype=tf.float32)
            relevant_score = tf.where(top_k_score >= self.threshold, ones, zeros)

            legal_atts = []
            dense_layer = tf.keras.layers.Dense(
                self.att_size,
                tf.nn.tanh,
                use_bias=False,
                kernel_regularizer=self.regularizer
            )
            for i in range(self.top_k):
                art_enc = top_k_art_enc[i]
                art_len = top_k_art_len[:, i]
                score = relevant_score[:, i]

                query = dense_layer(art_enc)
                att_matrix = tf.reshape(score, [-1, 1, 1]) * self.get_attention(query, key, art_len, self.fact_len)
                legal_atts.append(tf.reduce_sum(att_matrix, axis=-2))

            # prevent dividing by zero
            att_num = tf.reshape(tf.reduce_sum(relevant_score, axis=-1), [-1, 1])
            ones = tf.ones_like(att_num, dtype=tf.float32)
            att_num = tf.where(att_num > 0, att_num, ones)

            fact_enc_with_att = [tf.reduce_max(tf.expand_dims(att, axis=-1) * fact_enc, axis=-2) for att in legal_atts]
            fact_enc_with_att = tf.add_n(fact_enc_with_att) / att_num + tf.reduce_max(fact_enc, axis=-2)

        with tf.variable_scope('output'):
            self.task_1_output, task_1_loss = self.output_layer(fact_enc_with_att, self.accu, layer='sigmoid')

            ones = tf.ones_like(art_score, dtype=tf.float32)
            zeros = tf.zeros_like(art_score, dtype=tf.float32)
            self.task_2_output = tf.where(tf.nn.sigmoid(art_score) >= self.threshold, ones, zeros)

        with tf.variable_scope('loss'):
            task_2_loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.relevant_art, logits=art_score),
                axis=-1
            ))

            self.loss = task_1_loss + task_2_loss
            if self.regularizer is not None:
                l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                self.loss += l2_loss

        if not is_training:
            return

        self.global_step, self.train_op = self.get_train_op()

    def embedding_layer(self, inputs):
        inputs_em = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        if self.is_training and self.dropout < 1.0:
            inputs_em = tf.nn.dropout(inputs_em, rate=self.dropout)

        return inputs_em

    def cnn_encoder(self, inputs):
        enc_output = []
        for kernel_size in self.kernel_size:
            conv = tf.keras.layers.Conv1D(
                self.filter_dim,
                kernel_size,
                padding='same',
                kernel_regularizer=self.regularizer,
                name='conv_' + str(kernel_size)
            )(inputs)
            if self.use_batch_norm:
                conv = tf.keras.layers.BatchNormalization(name='norm_' + str(kernel_size))(conv)
            conv = tf.nn.relu(conv)
            enc_output.append(conv)

        enc_output = tf.concat(enc_output, axis=-1)

        return enc_output

    def get_top_k_indices(self, inputs):
        inputs = tf.reduce_max(inputs, axis=-2)
        scores = tf.keras.layers.Dense(self.art_num, kernel_regularizer=self.regularizer)(inputs)

        if self.is_training:
            top_k_score, top_k_indices = tf.math.top_k(self.relevant_art, k=self.top_k)
        else:
            top_k_score, top_k_indices = tf.math.top_k(tf.nn.sigmoid(scores), k=self.top_k)

        return scores, top_k_score, top_k_indices

    def get_top_k_articles(self, top_k_indices):
        top_k_art = tf.batch_gather(self.art, indices=top_k_indices)
        top_k_art_len = tf.batch_gather(self.art_len, indices=top_k_indices)

        return top_k_art, top_k_art_len

    def art_encoder(self, top_k_art_em, shared_layers):
        top_k_art_enc = []
        for i in range(self.top_k):
            art_enc = []
            art_em = top_k_art_em[:, i, :, :]
            for kernel_size in self.kernel_size:
                conv = shared_layers['conv_' + str(kernel_size)](art_em)
                if self.use_batch_norm:
                    conv = shared_layers['norm_' + str(kernel_size)](conv)
                conv = tf.nn.relu(conv)
                art_enc.append(conv)

            art_enc = tf.concat(art_enc, axis=-1)
            top_k_art_enc.append(art_enc)

        return top_k_art_enc

    def get_attention(self, query, key, query_len, key_len):
        att = tf.matmul(query, key, transpose_b=True)

        query_mask = tf.sequence_mask(query_len, maxlen=self.max_seq_len, dtype=tf.float32)
        key_mask = tf.sequence_mask(key_len, maxlen=self.max_seq_len, dtype=tf.float32)
        mask = tf.matmul(tf.expand_dims(query_mask, axis=-1), tf.expand_dims(key_mask, axis=-2))
        inf = 1e10 * tf.ones_like(att, dtype=tf.float32)

        # set masked value to -inf to prevent gradient
        masked_att = tf.where(mask > 0.0, att, -inf)
        masked_att = tf.nn.softmax(masked_att, axis=-1)
        masked_att = mask * masked_att

        return masked_att

    def output_layer(self, inputs, labels, layer):
        fc_output = tf.keras.layers.Dense(self.fc_size, kernel_regularizer=self.regularizer)(inputs)
        if self.is_training and self.dropout < 1.0:
            fc_output = tf.nn.dropout(fc_output, rate=self.dropout)

        logits = tf.keras.layers.Dense(labels.shape[-1], kernel_regularizer=self.regularizer)(fc_output)
        if layer == 'softmax':
            output = tf.nn.softmax(logits)
            ce_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
        elif layer == 'sigmoid':
            output = tf.nn.sigmoid(logits)
            ce_loss = tf.reduce_mean(tf.reduce_sum(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits),
                axis=-1
            ))
        else:
            assert False

        return output, ce_loss

    def get_train_op(self):
        global_step = tf.Variable(0, trainable=False, name='global_step')

        if self.optimizer == 'Adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'Adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'Adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        else:
            assert False

        train_op = optimizer.minimize(self.loss, global_step=global_step)

        return global_step, train_op
