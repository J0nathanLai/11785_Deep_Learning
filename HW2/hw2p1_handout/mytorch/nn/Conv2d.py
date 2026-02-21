import numpy as np
from resampling import *


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size, weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A

        output_height = A.shape[2] - self.kernel_size + 1
        output_width = A.shape[3] - self.kernel_size + 1
        Z = np.zeros((A.shape[0], self.out_channels, output_height, output_width))
        for i in range(output_height):
            for j in range(output_width):
                Z[:, :, i, j] = np.tensordot(A[:, :, i:i+self.kernel_size, j:j+self.kernel_size], self.W, axes=([1, 2, 3], [1, 2, 3]))
        Z += self.b[None, :, None, None]
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        self.dLdW = np.zeros(self.W.shape)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                self.dLdW[:, :, i, j] = np.tensordot(self.A[:, :, i:i+dLdZ.shape[2], j:j+dLdZ.shape[3]], dLdZ, axes=([0, 2, 3],[0, 2, 3])).T
        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))  # sum over batch, height and width
        pad_dLdZ = np.pad(dLdZ, ((0, 0), (0, 0), (self.kernel_size-1, self.kernel_size-1), (self.kernel_size-1, self.kernel_size-1)))
        flip_W = np.flip(self.W, axis=(2, 3))
        dLdA = np.zeros(self.A.shape)
        for i in range(self.A.shape[2]):
            for j in range(self.A.shape[3]):
                dLdA[:, :, i, j] = np.tensordot(
                    pad_dLdZ[:, :, i:i+self.kernel_size, j:j+self.kernel_size],
                    flip_W,
                    axes=([1, 2, 3],[0, 2, 3])
                )
        return dLdA


class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride
        self.pad = padding

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        # Pad the input appropriately using np.pad() function
        # TODO
        pad_A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)))

        # Call Conv2d_stride1
        # TODO
        conv_Z = self.conv2d_stride1.forward(pad_A)

        # downsample
        Z = self.downsample2d.forward(conv_Z) # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        # Call downsample1d backward
        # TODO
        dLdA_downsample = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA_downsample)  # TODO

        # Unpad the gradient
        # TODO
        if self.pad == 0:
            return dLdA
        unpad_dLdA = dLdA[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return unpad_dLdA
