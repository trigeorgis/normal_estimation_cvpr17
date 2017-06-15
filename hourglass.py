import tensorflow as tf
slim = tf.contrib.slim

def hourglass_arg_scope(weight_decay=0.0005,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                return arg_sc

def convolutional_block(num_in, num_out):
    def wrapper(inputs):
        net = slim.batch_norm(inputs)
        net = tf.nn.relu(net)
        net = slim.conv2d(net, num_out // 2, 1, normalizer_fn=slim.batch_norm)
        net = slim.conv2d(net, num_out // 2, 3, normalizer_fn=slim.batch_norm)
        net = slim.conv2d(net, num_out, 1, activation_fn=None)
        return net

    return wrapper

        
def skip_layer(num_in, num_out):
    def wrapper(inputs):
        if num_in == num_out:
            return inputs
        
        return slim.conv2d(inputs, num_out, 1)

    return wrapper

        
def residual(num_in, num_out):
    def wrapper(inputs):
        net = convolutional_block(num_in, num_out)(inputs)
        skip = skip_layer(num_in, num_out)(inputs)
        
        return net + skip

    return wrapper
        
def lin(inputs, num_out):
    return slim.conv2d(inputs, num_out, 1, 1, normalizer_fn=slim.batch_norm, padding='VALID')

def hourglass_module(inputs, n, num_features):
    # upper branch
    
    up1 = inputs
    up1 = residual(num_features, num_features)(up1)
     
    # lower branch
    low1 = slim.max_pool2d(inputs, 2)
    low1 = residual(num_features, num_features)(low1)

    if n > 1:
        low2 = hourglass_module(low1, n-1, num_features)
    else:
        low2 = residual(num_features, num_features)(low1)
    
    low3 = low2
    low3 = residual(num_features, num_features)(low3)
    up2 = tf.image.resize_bilinear(low3, tf.shape(inputs)[1:3], name="up_sample")

    return up1 + up2

def hourglass(inputs, num_outputs=3, num_features=256, num_stacked_hg=1):
    
    with slim.arg_scope(hourglass_arg_scope()):
        cnv1 = slim.conv2d(inputs, 
                           64, 7, 2, scope='conv1', 
                           normalizer_fn=slim.batch_norm)
        
        r1 = residual(64, 128)(cnv1)
        pool = slim.max_pool2d(r1, 2)
        r4 = residual(128, 128)(pool)
        
        small_num = 32
        r5 = residual(128, num_features)(r4)
        
        # inter = tf.image.resize_bilinear(r5, tf.shape(inputs)[1:3], name="up_sample")
        inter = r5
        
        for i in range(num_stacked_hg):
            
            hg = hourglass_module(inter, 4, num_features)
            ll = hg
            ll = residual(num_features, num_features)(ll)
            ll = lin(ll, num_features)
            
            prediction = slim.conv2d(ll, num_outputs, 1, activation_fn=None)
            
            if i < num_stacked_hg - 1:
                net = slim.conv2d(ll, num_features, 1, activation_fn=None)
                out = slim.conv2d(prediction, num_features, 1, activation_fn=None)
                
                inter = net + out

        net = tf.image.resize_bilinear(prediction, tf.shape(inputs)[1:3], name="up_sample")

        outputs = convolutional_block(num_outputs, num_outputs)(net)


    return outputs
