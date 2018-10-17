import tensorflow as tf


def multi_kernel_conv2d(inputs, num_outputs_each, kernel_sizes=[3, 5, 7], stride=1, padding='SAME',
                        activation_fn=tf.nn.leaky_relu, normalizer_fn=None, normalizer_params=None,
                        weights_initializer=tf.truncated_normal_initializer, weights_regularizer=None,
                        biases_initializer=tf.zeros_initializer, biases_regularizer=None,
                        reuse=None, scope='multi_kernel_conv'):
    """ This function performs convolution over the same input using different size kernels, the result is later concatenated.
        Args:
            inputs: A Tensor of rank N+2 of shape `[batch_size] + input_spatial_shape + [in_channels]`
            num_outputs_each: Integer, the number of output filters from each kernel size
            kernel_size: A sequence of N positive integers specifying the spatial dimensions of the filters (KxK) is the kernel size used
            stride: A sequence of N positive integers specifying the stride at which to compute output.  Can be a single integer to specify the same value for all spatial dimensions.  Specifying any `stride` value != 1 is incompatible with specifying any `rate` value != 1.
            padding: One of `"VALID"` or `"SAME"`.
            activation_fn: Activation function. The default value is a Leaky ReLU function. Explicitly set it to None to skip it and maintain a linear activation.
            normalizer_fn: Normalization function to use instead of `biases`. If `normalizer_fn` is provided then `biases_initializer` and `biases_regularizer` are ignored and `biases` are not created nor added. Default set to None for no normalizer function
            normalizer_params: Normalization function parameters.
            weights_initializer: An initializer for the weights.
            weights_regularizer: Optional regularizer for the weights.
            biases_initializer: An initializer for the biases. If None skip biases.
            biases_regularizer: Optional regularizer for the biases.
            reuse: Whether or not the layer and its variables should be reused. To be able to reuse the layer scope must be given.
            scope: Optional scope for `variable_scope`.

        Returns:
            A tensor representing the output of the operation.
    """

    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse()

        n = inputs.get_shape()[-1]
        assert type(kernel_sizes) is list, 'kernel sizes is not a list'

        convs = []
        for k in kernel_sizes:
            w = tf.get_variable('weights{}'.format(k), shape=[k, k] + [n, num_outputs_each], initializer=weights_initializer, regularizer=weights_regularizer)
            c = tf.nn.conv2d(inputs, w, [1, stride, stride, 1], padding)
            convs.append(c)
        c = tf.concat(convs, axis=-1)

        if normalizer_fn:
            c = normalizer_fn(c, **normalizer_params)
        else:
            b = tf.get_variable('biases', shape=[num_outputs_each * len(kernel_sizes)], initializer=biases_initializer, regularizer=biases_regularizer)
            c = c + b

        if activation_fn:
            c = activation_fn(c)

        return c


def multi_kernel_conv2d_transpose(inputs, num_outputs_each, kernel_sizes=[3, 5, 7], stride=1, padding='SAME',
                                  activation_fn=tf.nn.leaky_relu, normalizer_fn=None, normalizer_params=None,
                                  weights_initializer=tf.truncated_normal_initializer, weights_regularizer=None,
                                  biases_initializer=tf.zeros_initializer, biases_regularizer=None,
                                  reuse=None, scope='multi_kernel_conv'):
    """ This function performs deconvolution over the same input using different size kernels, the result is later concatenated.
        Args:
            inputs: A Tensor of rank N+2 of shape `[batch_size] + input_spatial_shape + [in_channels]`
            num_outputs_each: Integer, the number of output filters from each kernel size
            kernel_size: A sequence of N positive integers specifying the spatial dimensions of the filters (KxK) is the kernel size used
            stride: A sequence of N positive integers specifying the stride at which to compute output.  Can be a single integer to specify the same value for all spatial dimensions.  Specifying any `stride` value != 1 is incompatible with specifying any `rate` value != 1.
            padding: One of `"VALID"` or `"SAME"`.
            activation_fn: Activation function. The default value is a Leaky ReLU function. Explicitly set it to None to skip it and maintain a linear activation.
            normalizer_fn: Normalization function to use instead of `biases`. If `normalizer_fn` is provided then `biases_initializer` and `biases_regularizer` are ignored and `biases` are not created nor added. Default set to None for no normalizer function
            normalizer_params: Normalization function parameters.
            weights_initializer: An initializer for the weights.
            weights_regularizer: Optional regularizer for the weights.
            biases_initializer: An initializer for the biases. If None skip biases.
            biases_regularizer: Optional regularizer for the biases.
            reuse: Whether or not the layer and its variables should be reused. To be able to reuse the layer scope must be given.
            scope: Optional scope for `variable_scope`.

        Returns:
            A tensor representing the output of the operation.
    """
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse()

        assert type(kernel_sizes) is list, 'kernel sizes is not a list'

        convs = []
        for i, k in enumerate(kernel_sizes):
            c = tf.contrib.layers.conv2d_transpose(inputs, num_outputs_each, k, stride=stride, activation_fn=None, scope='c{}'.format(i))
            convs.append(c)
        c = tf.concat(convs, axis=-1)

        if normalizer_fn:
            c = normalizer_fn(c, **normalizer_params)
        else:
            b = tf.get_variable('biases', shape=[num_outputs_each * len(kernel_sizes)], initializer=biases_initializer, regularizer=biases_regularizer)
            c = c + b

        if activation_fn:
            c = activation_fn(c)

        return c


def resnet_block_transpose(inputs, num_outputs, kernel_size=3, stride=1, activation_fn=None, reuse=None, scope='resnet_block'):
    """ This function performs deconvolution over the input by resizing the input and adding a computed residual to it.
        Args:
            inputs: A Tensor of rank N+2 of shape `[batch_size] + input_spatial_shape + [in_channels]`
            num_outputs: Integer, the number of output filters from each kernel size
            kernel_size: A sequence of N positive integers specifying the spatial dimensions of the filters, (KxK) is the kernel size used
            stride: A sequence of N positive integers specifying the stride at which to compute output.  Can be a single integer to specify the same value for all spatial dimensions.  Specifying any `stride` value != 1 is incompatible with specifying any `rate` value != 1.
            activation_fn: Activation function. The default value is a Leaky ReLU function. Explicitly set it to None to skip it and maintain a linear activation.
            reuse: Whether or not the layer and its variables should be reused. To be able to reuse the layer scope must be given.
            scope: Optional scope for `variable_scope`.

        Returns:
            A tensor representing the output of the operation.
    """
    with tf.variable_scope(scope):
        if reuse:
            tf.get_variable_scope().reuse()

        ## It was observed that the model performed better without any regularizer
        norm_fn = None
        norm_params = {}

        shape = inputs.get_shape()
        h = int(shape[1])
        w = int(shape[2])
        c = int(shape[3])
        ## First change the output to required number of channels with Linear activation
        c0 = tf.contrib.layers.conv2d(inputs, num_outputs, 1, stride=1, activation_fn=None, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='c0')
        ## Resize the input to required size, calulated using the given stride
        c1 = tf.image.resize_images(c0, [h * stride, w * stride])
        ## Perform a convolution over the original input with equal number of channels, kernel_size=1 and a leaky relu activation
        r1 = tf.contrib.layers.conv2d(inputs, c, kernel_size=1, stride=1, activation_fn=tf.nn.leaky_relu, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='r1')
        ## Calculate the residual using a second convolution with no activation and required number of channels
        r2 = tf.contrib.layers.conv2d_transpose(r1, num_outputs, kernel_size, stride=stride, activation_fn=None, normalizer_fn=norm_fn, normalizer_params=norm_params, scope='r2')
        ## Add the residual to the resized input to get the output
        output = c1 + r2
        ## Apply any activation if required
        if activation_fn is not None:
            output = activation_fn(output)
        return output


# End
