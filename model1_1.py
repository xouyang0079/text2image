# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
from Utils import ops

class GAN:
    '''
    OPTIONS
    z_dim : Noise dimension 100
    t_dim : Text feature dimension 256
    image_size : Image Dimension 64
    gf_dim : Number of conv in the first layer generator 64
    df_dim : Number of conv in the first layer discriminator 64
    gfc_dim : Dimension of gen untis for for fully connected layer 1024
    caption_vector_length : Caption Vector Length 2400
    batch_size : Batch Size 64
    '''
    def __init__(self, options):
        self.options = options

        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')

        self.d_bn1 = ops.batch_norm(name='d_bn1')
        self.d_bn2 = ops.batch_norm(name='d_bn2')
        self.d_bn3 = ops.batch_norm(name='d_bn3')
        self.d_bn4 = ops.batch_norm(name='d_bn4')


    def build_model(self):
        img_size = self.options['image_size']
        t_real_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'real_image')
        t_wrong_image = tf.placeholder('float32', [self.options['batch_size'],img_size, img_size, 3 ], name = 'wrong_image')
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']], name = 'real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], 15, 400], name = 't_z')
        mask_x0 = tf.placeholder('float32', [self.options['batch_size'], 15, 400], name = 'mask')
        mask_x = tf.transpose(mask_x0, perm=[1, 0, 2])
        mask0_1 = tf.reduce_mean(mask_x0, 2)  # batchsize*length
        mask1 = mask0_1[:, :, None]  # batchsize*length*1
        mask2 = tf.expand_dims(mask1, 3)
        mask2 = tf.expand_dims(mask2, 4)
        mask2 = tf.tile(mask2, [1,1,64,64,3]) # batchsize*length*64*64*3

        # build LSTM network
        hidden_neural_size = 400
        hidden_layer_num = 1
        num_step = 15

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_neural_size, forget_bias=0.0, state_is_tuple=True)

        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * hidden_layer_num, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.options['batch_size'], dtype=tf.float32)

        out_put = []
        state = self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(num_step):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(t_z[:, time_step, :], state)
                out_put.append(cell_output)

        out_put = out_put * mask_x
        proj = tf.transpose(out_put, perm=[1, 0, 2])
        
        # build generator
        proj0 = proj[:, 0:1, :]
        proj0 = tf.reshape(proj0, (self.options['batch_size'], 400))  # batchsize*400
        proj1 = proj[:, 1:2, :]
        proj1 = tf.reshape(proj1, (self.options['batch_size'], 400))  # batchsize*400
        proj2 = proj[:, 2:3, :]
        proj2 = tf.reshape(proj2, (self.options['batch_size'], 400))  # batchsize*400
        proj3 = proj[:, 3:4, :]
        proj3 = tf.reshape(proj3, (self.options['batch_size'], 400))  # batchsize*400
        proj4 = proj[:, 4:5, :]
        proj4 = tf.reshape(proj4, (self.options['batch_size'], 400))  # batchsize*400
        proj5 = proj[:, 5:6, :]
        proj5 = tf.reshape(proj5, (self.options['batch_size'], 400))  # batchsize*400
        proj6 = proj[:, 6:7, :]
        proj6 = tf.reshape(proj6, (self.options['batch_size'], 400))  # batchsize*400
        proj7 = proj[:, 7:8, :]
        proj7 = tf.reshape(proj7, (self.options['batch_size'], 400))  # batchsize*400
        proj8 = proj[:, 8:9, :]
        proj8 = tf.reshape(proj8, (self.options['batch_size'], 400))  # batchsize*400
        proj9 = proj[:, 9:10, :]
        proj9 = tf.reshape(proj9, (self.options['batch_size'], 400))  # batchsize*400
        proj10 = proj[:, 10:11, :]
        proj10 = tf.reshape(proj10, (self.options['batch_size'], 400))  # batchsize*400
        proj11 = proj[:, 11:12, :]
        proj11 = tf.reshape(proj11, (self.options['batch_size'], 400))  # batchsize*400
        proj12 = proj[:, 12:13, :]
        proj12 = tf.reshape(proj12, (self.options['batch_size'], 400))  # batchsize*400
        proj13 = proj[:, 13:14, :]
        proj13 = tf.reshape(proj13, (self.options['batch_size'], 400))  # batchsize*400
        proj14 = proj[:, 14:15, :]
        proj14 = tf.reshape(proj14, (self.options['batch_size'], 400))  # batchsize*400

        proj_new0 = self.generator(proj0)[0]
        proj_new1 = self.generator(proj1, reuse=True)[0]
        proj_new2 = self.generator(proj2, reuse=True)[0]
        proj_new3 = self.generator(proj3, reuse=True)[0]
        proj_new4 = self.generator(proj4, reuse=True)[0]
        proj_new5 = self.generator(proj5, reuse=True)[0]
        proj_new6 = self.generator(proj6, reuse=True)[0]
        proj_new7 = self.generator(proj7, reuse=True)[0]
        proj_new8 = self.generator(proj8, reuse=True)[0]
        proj_new9 = self.generator(proj9, reuse=True)[0]
        proj_new10 = self.generator(proj10, reuse=True)[0]
        proj_new11 = self.generator(proj11, reuse=True)[0]
        proj_new12 = self.generator(proj12, reuse=True)[0]
        proj_new13 = self.generator(proj13, reuse=True)[0]
        proj_new14 = self.generator(proj14, reuse=True)[0]

        
        fake_image_new = tf.concat(
            [proj_new0[:, None, :, :, :],proj_new1[:, None, :, :, :],proj_new2[:, None, :, :, :],
             proj_new3[:, None, :, :, :],proj_new4[:, None, :, :, :],
             proj_new5[:, None, :, :, :],proj_new6[:, None, :, :, :],proj_new7[:, None, :, :, :],
             proj_new8[:, None, :, :, :],proj_new9[:, None, :, :, :],
             proj_new10[:, None, :, :, :],proj_new11[:, None, :, :, :],proj_new12[:, None, :, :, :],
             proj_new13[:, None, :, :, :],proj_new14[:, None, :, :, :]], 1)

        fake_image_new = fake_image_new * mask2
        
        mask_new0 = self.generator(proj0, reuse=True)[1]
        mask_new1 = self.generator(proj1, reuse=True)[1]
        mask_new2 = self.generator(proj2, reuse=True)[1]
        mask_new3 = self.generator(proj3, reuse=True)[1]
        mask_new4 = self.generator(proj4, reuse=True)[1]
        mask_new5 = self.generator(proj5, reuse=True)[1]
        mask_new6 = self.generator(proj6, reuse=True)[1]
        mask_new7 = self.generator(proj7, reuse=True)[1]
        mask_new8 = self.generator(proj8, reuse=True)[1]
        mask_new9 = self.generator(proj9, reuse=True)[1]
        mask_new10 = self.generator(proj10, reuse=True)[1]
        mask_new11 = self.generator(proj11, reuse=True)[1]
        mask_new12 = self.generator(proj12, reuse=True)[1]
        mask_new13 = self.generator(proj13, reuse=True)[1]
        mask_new14 = self.generator(proj14, reuse=True)[1]

        
        mask_new = tf.concat(
            [mask_new0[:, None, :, :, :],mask_new1[:, None, :, :, :],mask_new2[:, None, :, :, :],
             mask_new3[:, None, :, :, :],mask_new4[:, None, :, :, :],
             mask_new5[:, None, :, :, :],mask_new6[:, None, :, :, :],mask_new7[:, None, :, :, :],
             mask_new8[:, None, :, :, :],mask_new9[:, None, :, :, :],
             mask_new10[:, None, :, :, :],mask_new11[:, None, :, :, :],mask_new12[:, None, :, :, :],
             mask_new13[:, None, :, :, :],mask_new14[:, None, :, :, :]], 1)

        mask_new = mask_new * mask2

        disc_fake_image0, disc_fake_image_logits0 = self.discriminator(proj_new0, t_real_caption)
        disc_fake_image1, disc_fake_image_logits1 = self.discriminator(proj_new1, t_real_caption, reuse = True)
        disc_fake_image2, disc_fake_image_logits2 = self.discriminator(proj_new2, t_real_caption, reuse = True)
        disc_fake_image3, disc_fake_image_logits3 = self.discriminator(proj_new3, t_real_caption, reuse = True)
        disc_fake_image4, disc_fake_image_logits4 = self.discriminator(proj_new4, t_real_caption, reuse = True)
        disc_fake_image5, disc_fake_image_logits5 = self.discriminator(proj_new5, t_real_caption, reuse = True)
        disc_fake_image6, disc_fake_image_logits6 = self.discriminator(proj_new6, t_real_caption, reuse = True)
        disc_fake_image7, disc_fake_image_logits7 = self.discriminator(proj_new7, t_real_caption, reuse = True)
        disc_fake_image8, disc_fake_image_logits8 = self.discriminator(proj_new8, t_real_caption, reuse = True)
        disc_fake_image9, disc_fake_image_logits9 = self.discriminator(proj_new9, t_real_caption, reuse = True)
        disc_fake_image10, disc_fake_image_logits10 = self.discriminator(proj_new10, t_real_caption, reuse = True)
        disc_fake_image11, disc_fake_image_logits11 = self.discriminator(proj_new11, t_real_caption, reuse = True)
        disc_fake_image12, disc_fake_image_logits12 = self.discriminator(proj_new12, t_real_caption, reuse = True)
        disc_fake_image13, disc_fake_image_logits13 = self.discriminator(proj_new13, t_real_caption, reuse = True)
        disc_fake_image14, disc_fake_image_logits14 = self.discriminator(proj_new14, t_real_caption, reuse = True)


        disc_fake_image_new = tf.concat(
            [disc_fake_image0[:, None, :],disc_fake_image1[:,None,:],disc_fake_image2[:,None,:],
             disc_fake_image3[:,None,:],disc_fake_image4[:,None,:],
             disc_fake_image5[:, None, :],disc_fake_image6[:,None,:],disc_fake_image7[:,None,:],
             disc_fake_image8[:,None,:],disc_fake_image9[:,None,:],
             disc_fake_image10[:, None, :],disc_fake_image11[:,None,:],disc_fake_image12[:,None,:],
             disc_fake_image13[:,None,:],disc_fake_image14[:,None,:]], 1)


        disc_fake_image_logits_new = tf.concat(
            [disc_fake_image_logits0[:, None, :], disc_fake_image_logits1[:, None, :], disc_fake_image_logits2[:, None, :],
             disc_fake_image_logits3[:, None, :], disc_fake_image_logits4[:, None, :],
             disc_fake_image_logits5[:, None, :], disc_fake_image_logits6[:, None, :], disc_fake_image_logits7[:, None, :],
             disc_fake_image_logits8[:, None, :], disc_fake_image_logits9[:, None, :],
             disc_fake_image_logits10[:, None, :], disc_fake_image_logits11[:, None, :], disc_fake_image_logits12[:, None, :],
             disc_fake_image_logits13[:, None, :], disc_fake_image_logits14[:, None, :]], 1)


        disc_real_image0, disc_real_image_logits0 = self.discriminator(t_real_image*mask_new0, t_real_caption, reuse = True)
        disc_real_image1, disc_real_image_logits1 = self.discriminator(t_real_image*mask_new1, t_real_caption, reuse = True)
        disc_real_image2, disc_real_image_logits2 = self.discriminator(t_real_image*mask_new2, t_real_caption, reuse = True)
        disc_real_image3, disc_real_image_logits3 = self.discriminator(t_real_image*mask_new3, t_real_caption, reuse = True)
        disc_real_image4, disc_real_image_logits4 = self.discriminator(t_real_image*mask_new4, t_real_caption, reuse = True)
        disc_real_image5, disc_real_image_logits5 = self.discriminator(t_real_image*mask_new5, t_real_caption, reuse = True)
        disc_real_image6, disc_real_image_logits6 = self.discriminator(t_real_image*mask_new6, t_real_caption, reuse = True)
        disc_real_image7, disc_real_image_logits7 = self.discriminator(t_real_image*mask_new7, t_real_caption, reuse = True)
        disc_real_image8, disc_real_image_logits8 = self.discriminator(t_real_image*mask_new8, t_real_caption, reuse = True)
        disc_real_image9, disc_real_image_logits9 = self.discriminator(t_real_image*mask_new9, t_real_caption, reuse = True)
        disc_real_image10, disc_real_image_logits10 = self.discriminator(t_real_image*mask_new10, t_real_caption, reuse = True)
        disc_real_image11, disc_real_image_logits11 = self.discriminator(t_real_image*mask_new11, t_real_caption, reuse = True)
        disc_real_image12, disc_real_image_logits12 = self.discriminator(t_real_image*mask_new12, t_real_caption, reuse = True)
        disc_real_image13, disc_real_image_logits13 = self.discriminator(t_real_image*mask_new13, t_real_caption, reuse = True)
        disc_real_image14, disc_real_image_logits14 = self.discriminator(t_real_image*mask_new14, t_real_caption, reuse = True)

        disc_real_image_new = tf.concat(
            [disc_real_image0[:, None, :], disc_real_image1[:, None, :], disc_real_image2[:, None, :],
             disc_real_image3[:, None, :], disc_real_image4[:, None, :],
             disc_real_image5[:, None, :], disc_real_image6[:, None, :], disc_real_image7[:, None, :],
             disc_real_image8[:, None, :], disc_real_image9[:, None, :],
             disc_real_image10[:, None, :], disc_real_image11[:, None, :], disc_real_image12[:, None, :],
             disc_real_image13[:, None, :],
             disc_real_image14[:, None, :]], 1)

        disc_real_image_logits_new = tf.concat(
            [disc_real_image_logits0[:, None, :], disc_real_image_logits1[:, None, :],
             disc_real_image_logits2[:, None, :],
             disc_real_image_logits3[:, None, :], disc_real_image_logits4[:, None, :],
             disc_real_image_logits5[:, None, :], disc_real_image_logits6[:, None, :],
             disc_real_image_logits7[:, None, :],
             disc_real_image_logits8[:, None, :], disc_real_image_logits9[:, None, :],
             disc_real_image_logits10[:, None, :], disc_real_image_logits11[:, None, :],
             disc_real_image_logits12[:, None, :],
             disc_real_image_logits13[:, None, :], disc_real_image_logits14[:, None, :]], 1)
        
        disc_wrong_image, disc_wrong_image_logits = self.discriminator(t_wrong_image, t_real_caption, reuse = True)
        disc_wrong_image_new = tf.concat(
            [disc_wrong_image[:, None, :], disc_wrong_image[:, None, :], disc_wrong_image[:, None, :],
             disc_wrong_image[:, None, :], disc_wrong_image[:, None, :],
             disc_wrong_image[:, None, :], disc_wrong_image[:, None, :], disc_wrong_image[:, None, :],
             disc_wrong_image[:, None, :], disc_wrong_image[:, None, :],
             disc_wrong_image[:, None, :], disc_wrong_image[:, None, :], disc_wrong_image[:, None, :],
             disc_wrong_image[:, None, :], disc_wrong_image[:, None, :]], 1)

        disc_wrong_image_logits_new = tf.concat(
            [disc_wrong_image_logits[:, None, :], disc_wrong_image_logits[:, None, :],
             disc_wrong_image_logits[:, None, :],
             disc_wrong_image_logits[:, None, :], disc_wrong_image_logits[:, None, :],
             disc_wrong_image_logits[:, None, :], disc_wrong_image_logits[:, None, :],
             disc_wrong_image_logits[:, None, :],
             disc_wrong_image_logits[:, None, :], disc_wrong_image_logits[:, None, :],
             disc_wrong_image_logits[:, None, :], disc_wrong_image_logits[:, None, :],
             disc_wrong_image_logits[:, None, :],
             disc_wrong_image_logits[:, None, :], disc_wrong_image_logits[:, None, :]], 1)

        g_loss = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_fake_image_new), logits=disc_fake_image_logits_new))*mask1)
        d_loss1 = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(disc_real_image_new), logits=disc_real_image_logits_new))*mask1)
        d_loss2 = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_wrong_image_new), logits=disc_wrong_image_logits_new))*mask1)
        d_loss3 = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(disc_fake_image_new), logits=disc_fake_image_logits_new))*mask1)

        d_loss = d_loss1 + d_loss2 + d_loss3

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        g1_vars = [var for var in t_vars if 'g1_' in var.name]
        l_vars = [var for var in t_vars if 'LSTM' in var.name]

        input_tensors = {
            't_real_image' : t_real_image,
            't_wrong_image' : t_wrong_image,
            't_real_caption' : t_real_caption,
            't_z' : t_z,
            'mask': mask_x0
        }

        variables = {
            'd_vars' : d_vars,
            'g_vars' : g_vars,
            'g1_vars': g1_vars,
            'l_vars': l_vars
        }

        loss = {
            'g_loss' : g_loss,
            'd_loss' : d_loss
        }

        outputs = {
            'generator' : fake_image_new,
            'mask_out' : mask_new
        }

        checks = {
            'd_loss1': d_loss1,
            'd_loss2': d_loss2,
            'd_loss3' : d_loss3,
            'disc_real_image_logits' : disc_real_image_logits_new,
            'disc_wrong_image_logits' : disc_wrong_image_new,
            'disc_fake_image_logits' : disc_fake_image_logits_new
        }

        return input_tensors, variables, loss, outputs, checks


    # Sample Images for a txt embedding
    def sampler(self, t_text_embedding):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s = self.options['image_size']
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # reduced_text_embedding = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
            # z_concat = tf.concat(1, [t_z, reduced_text_embedding])
            z_ = ops.linear(t_text_embedding, self.options['gf_dim'] * 8 * s16 * s16, 'g_h0_lin')
            h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train = False))

            h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim'] * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train = False))

            h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim'] * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train = False))

            h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim'] * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train = False))

            h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

            return (tf.tanh(h4)/2. + 0.5)

    # GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def generator(self, t_z, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()

            s = self.options['image_size']
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # reduced_text_embedding = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
            # #reduced_text_embedding = ops.lrelu(layers.fully_connected(t_text_embedding, self.options['t_dim'], name='g_embedding'))
            # z_concat = tf.concat([t_z, reduced_text_embedding], 1)
            z_ = ops.linear(t_z, self.options['gf_dim'] * 8 * s16 * s16, 'g_h0_lin')
            #z_ = layers.fully_connected(z_concat, self.options['gf_dim']*8*s16*s16, name='g_h0_lin')
            h0 = tf.reshape(z_, [-1, s16, s16, self.options['gf_dim'] * 8])
            h0 = tf.nn.relu(self.g_bn0(h0))

            h1 = ops.deconv2d(h0, [self.options['batch_size'], s8, s8, self.options['gf_dim'] * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))

            h2 = ops.deconv2d(h1, [self.options['batch_size'], s4, s4, self.options['gf_dim'] * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3 = ops.deconv2d(h2, [self.options['batch_size'], s2, s2, self.options['gf_dim'] * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 3], name='g_h4')

            h5 = ops.deconv2d(h3, [self.options['batch_size'], s, s, 1], name='g1_h5')

            h6 = tf.tile(h5, [1,1,1,3], name='g_h6')

            return (tf.tanh(h4)/2. + 0.5), (tf.tanh(h6)/2. + 0.5), (tf.tanh(h5)/2. + 0.5)

    # DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def discriminator(self, image, t_text_embedding, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = ops.lrelu(ops.conv2d(image, self.options['df_dim'], name ='d_h0_conv')) #32
            h1 = ops.lrelu(self.d_bn1(ops.conv2d(h0, self.options['df_dim'] * 2, name ='d_h1_conv'))) #16
            h2 = ops.lrelu(self.d_bn2(ops.conv2d(h1, self.options['df_dim'] * 4, name ='d_h2_conv'))) #8
            h3 = ops.lrelu(self.d_bn3(ops.conv2d(h2, self.options['df_dim'] * 8, name ='d_h3_conv'))) #4

            # ADD TEXT EMBEDDING TO THE NETWORK
            reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,1)
            reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings,2)
            tiled_embeddings = tf.tile(reduced_text_embeddings, [1,4,4,1], name='tiled_embeddings')

            h3_concat = tf.concat([h3, tiled_embeddings], 3, name='h3_concat')
            h3_new = ops.lrelu(self.d_bn4(
                ops.conv2d(h3_concat, self.options['df_dim'] * 8, 1, 1, 1, 1, name ='d_h3_conv_new'))) #4

            h4 = ops.linear(tf.reshape(h3_new, [self.options['batch_size'], -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4