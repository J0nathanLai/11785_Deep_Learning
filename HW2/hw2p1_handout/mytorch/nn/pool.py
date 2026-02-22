import numpy as np
from resampling import *


class MaxPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A_shape = A.shape
        output_width = A.shape[2] - self.kernel + 1
        output_height = A.shape[3] - self.kernel + 1
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        self.argmax_i = np.zeros(Z.shape, dtype=int)
        self.argmax_j = np.zeros(Z.shape, dtype=int)
        for i in range(output_width):
            for j in range(output_height):
                flat_patch = A[:, :, i:i+self.kernel, j:j+self.kernel].reshape(A.shape[0], A.shape[1], -1)
                Z[:, :, i, j] = np.max(flat_patch, axis=2)
                arg = np.argmax(flat_patch, axis=2)
                arg_i, arg_j = np.unravel_index(arg, (self.kernel, self.kernel))
                self.argmax_i[:, :, i, j] = i + arg_i
                self.argmax_j[:, :, i, j] = j + arg_j
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A_shape)
        for i in range(dLdZ.shape[2]):
            for j in range(dLdZ.shape[3]):
                arg_i = self.argmax_i[:, :, i, j]
                arg_j = self.argmax_j[:, :, i, j]
                dLdA[
                    np.arange(dLdZ.shape[0])[:, None], 
                    np.arange(dLdZ.shape[1])[None, :], 
                    arg_i, 
                    arg_j
                ] += dLdZ[:, :, i, j]
        return dLdA


class MeanPool2d_stride1():
    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A_shape = A.shape
        output_width = A.shape[2] - self.kernel + 1
        output_height = A.shape[3] - self.kernel + 1
        Z = np.zeros((A.shape[0], A.shape[1], output_width, output_height))
        for i in range(output_width):
            for j in range(output_height):
                Z[:, :, i, j] = np.mean(A[:, :, i:i+self.kernel, j:j+self.kernel], axis=(2, 3))
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A_shape)
        for i in range(dLdZ.shape[2]):
            for j in range(dLdZ.shape[3]):
                dLdA[:, :, i:i+self.kernel, j:j+self.kernel] += dLdZ[:, :, i, j][:, :, None, None] / (self.kernel**2)
        return dLdA


class MaxPool2d():
    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z_downsample = self.downsample2d.forward(Z)
        return Z_downsample

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_downsample = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ_downsample)
        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)  # TODO
        self.downsample2d = Downsample2d(self.stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z_downsample = self.downsample2d.forward(Z)
        return Z_downsample

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ_downsample = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ_downsample)
        return dLdA
