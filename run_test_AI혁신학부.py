import os
import sys
import pickle
import argparse
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = xp.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = xp.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = xp.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1), dtype=col.dtype)

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]


def shuffle_dataset(x, y_fine, y_coarse):
    permutation = np.random.permutation(x.shape[0])
    x_shuffled = x[permutation,:,:,:]
    y_fine_shuffled = y_fine[permutation]
    y_coarse_shuffled = y_coarse[permutation]

    return x_shuffled, y_fine_shuffled, y_coarse_shuffled


class SGD:
    """확률적 경사 하강법（Stochastic Gradient Descent）"""
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 
       
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = xp.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = xp.dot(dout, self.W.T)
        self.dW = xp.dot(self.x.T, dout)
        self.db = xp.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None 
        self.y = None    
        self.t = None    
        
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[xp.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            keep_prob = 1.0 - self.dropout_ratio
            self.mask = xp.random.rand(*x.shape) < keep_prob
            return (x * self.mask) / keep_prob
        else:
            return x

    def backward(self, dout):
        keep_prob = 1.0 - self.dropout_ratio
        return (dout * self.mask) / keep_prob
    
class BatchNormalization:
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        
        self.momentum = momentum
        
        self.input_shape = None

        self.running_mean = running_mean
        self.running_var = running_var  
        
        self.batch_size = None
        self.xc = None
        self.std = None
        self.xn = None 
        
        self.dgamma = None
        self.dbeta = None
        
    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        
        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        out = self.__forward(x, train_flg) 

        if len(self.input_shape) == 4:
            N, C, H, W = self.input_shape
            out = out.reshape(N, H, W, C)
            out = out.transpose(0, 3, 1, 2)

        return out

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = xp.zeros(D)
            self.running_var = xp.zeros(D)

        if train_flg:
            mu = x.mean(axis=0) 
            xc = x - mu
            var = xp.mean(xc**2, axis=0)
            std = xp.sqrt(var + 1e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn 
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / (xp.sqrt(self.running_var + 1e-7))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        input_shape_original = self.input_shape
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout = dout.transpose(0, 2, 3, 1).reshape(-1, C)

        dx = self.__backward(dout) 

        if len(input_shape_original) == 4:
            N, C, H, W = input_shape_original
            dx = dx.reshape(N, H, W, C)
            dx = dx.transpose(0, 3, 1, 2)

        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = xp.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -xp.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = xp.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        self.x = None   
        self.col = None
        self.col_W = None
        
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad) if GPU else im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        
        out = xp.dot(col, col_W)
        
        if self.b is not None:
            out += self.b
            
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        if self.b is not None:
            self.db = xp.sum(dout, axis=0)
            
        self.dW = xp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = xp.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad) 

        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - self.pool_h) / self.stride)
        out_w = int(1 + (W + 2*self.pad - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        
        pool_size = self.pool_h * self.pool_w
        col = col.reshape(-1, pool_size)

        self.arg_max = xp.argmax(col, axis=1)
        out = xp.max(col, axis=1)
        
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = xp.zeros((dout.size, pool_size))
        
        dmax[xp.arange(self.arg_max.size), self.arg_max] = dout.flatten()
        
        N, C, H, W = self.x.shape
        
        out_h = int(1 + (H + 2*self.pad - self.pool_h) / self.stride)
        out_w = int(1 + (W + 2*self.pad - self.pool_w) / self.stride)

        dcol = dmax.reshape(N, C, out_h, out_w, -1)
        dcol = dcol.transpose(0, 2, 3, 1, 4)
        dcol = dcol.reshape(N * out_h * out_w, -1)
        
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
    
def relu(x):
    return xp.maximum(0, x)

def softmax(x):
    x = x - xp.max(x, axis=-1, keepdims=True) 
    return xp.exp(x) / xp.sum(xp.exp(x), axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -xp.sum(xp.log(y[xp.arange(batch_size), t] + 1e-7)) / batch_size

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

class AdamW:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = xp.zeros_like(val)
                self.v[key] = xp.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * xp.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            update_term = lr_t * self.m[key] / (xp.sqrt(self.v[key]) + 1e-7)
            params[key] -= update_term
            
            if self.weight_decay > 0:
                params[key] -= self.lr * self.weight_decay * params[key]

class ResidualBlock:
    def __init__(self, all_params, param_keys, stride=1):
        self.param_keys = param_keys 

        self.conv1 = Convolution(
            W=all_params[param_keys['conv1_w']],
            b=all_params[param_keys['conv1_b']],
            stride=stride,
            pad=1
        )
        self.bn1 = BatchNormalization(
            gamma=all_params[param_keys['bn1_gamma']],
            beta=all_params[param_keys['bn1_beta']]
        )
        self.relu = Relu()
        self.conv2 = Convolution(
            W=all_params[param_keys['conv2_w']],
            b=all_params[param_keys['conv2_b']],
            stride=1,
            pad=1
        )
        self.bn2 = BatchNormalization(
            gamma=all_params[param_keys['bn2_gamma']],
            beta=all_params[param_keys['bn2_beta']]
        )

        self.shortcut = None
        if 'sc_w' in param_keys:
            self.shortcut = Convolution(
                W=all_params[param_keys['sc_w']],
                b=all_params[param_keys['sc_b']],
                stride=stride,
                pad=0
            )
            self.shortcut_bn = BatchNormalization(
                gamma=all_params[param_keys['sc_bn_gamma']],
                beta=all_params[param_keys['sc_bn_beta']]
            )
        
        self.layers = [self.conv1, self.bn1, self.relu, self.conv2, self.bn2]

    def forward(self, x, train_flg=True):
        identity = x
        out = x
        for layer in self.layers:
            if isinstance(layer, (BatchNormalization, Dropout)):
                out = layer.forward(out, train_flg)
            else:
                out = layer.forward(out)
        
        if self.shortcut:
            identity = self.shortcut.forward(identity)
            identity = self.shortcut_bn.forward(identity, train_flg)
        
        out += identity
        out = self.relu.forward(out) 
        
        return out

    def backward(self, dout):
        dout = self.relu.backward(dout) 
        
        d_identity = dout 
        d_main = dout

        if self.shortcut:
            d_identity = self.shortcut_bn.backward(d_identity)
            d_identity = self.shortcut.backward(d_identity)

        for layer in reversed(self.layers):
            d_main = layer.backward(d_main)
            
        return d_main + d_identity

class ResNet18:
    def __init__(self, input_dim=(3, 32, 32), output_size=100):
        """
        ResNet-18 클래스 초기화.
        [수정] 파라미터 생성 및 등록 방식을 일관되게 리팩토링하여 참조 오류를 원천 차단.
        """
        self.params = {}
        self.layers = OrderedDict()

        self.params['W1'] = xp.random.randn(64, input_dim[0], 3, 3) * xp.sqrt(2.0 / (input_dim[0] * 9))
        self.params['gamma1'] = xp.ones(64)
        self.params['beta1'] = xp.zeros(64)
        
        self.layers['conv1'] = Convolution(self.params['W1'], None, stride=1, pad=1)
        self.layers['bn1'] = BatchNormalization(gamma=self.params['gamma1'], beta=self.params['beta1'])
        self.layers['relu1'] = Relu()

        self.layers['conv2_x'] = self._make_layer(in_channels=64, out_channels=64, num_blocks=2, stride=1, stage_name='2')
        self.layers['conv3_x'] = self._make_layer(in_channels=64, out_channels=128, num_blocks=2, stride=2, stage_name='3')
        self.layers['conv4_x'] = self._make_layer(in_channels=128, out_channels=256, num_blocks=2, stride=2, stage_name='4')
        self.layers['conv5_x'] = self._make_layer(in_channels=256, out_channels=512, num_blocks=2, stride=2, stage_name='5')

        self.layers['avgpool'] = Pooling(pool_h=4, pool_w=4, stride=1, pad=0)
        
        self.params['W_affine'] = xp.random.randn(512, output_size) * xp.sqrt(2.0 / 512)
        self.params['b_affine'] = xp.zeros(output_size)
        self.layers['affine'] = Affine(self.params['W_affine'], self.params['b_affine'])
        
        self.last_layer = SoftmaxWithLoss()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, stage_name):
        
        layers = []
        
        param_keys_0 = {
            'conv1_w': f'W_{stage_name}_0_conv1', 'conv1_b': f'b_{stage_name}_0_conv1',
            'bn1_gamma': f'gamma_{stage_name}_0_bn1', 'bn1_beta': f'beta_{stage_name}_0_bn1',
            'conv2_w': f'W_{stage_name}_0_conv2', 'conv2_b': f'b_{stage_name}_0_conv2',
            'bn2_gamma': f'gamma_{stage_name}_0_bn2', 'bn2_beta': f'beta_{stage_name}_0_bn2',
        }
        self.params[param_keys_0['conv1_w']] = xp.random.randn(out_channels, in_channels, 3, 3) * xp.sqrt(2.0 / (in_channels * 9))
        self.params[param_keys_0['conv1_b']] = xp.zeros(out_channels)
        self.params[param_keys_0['bn1_gamma']] = xp.ones(out_channels)
        self.params[param_keys_0['bn1_beta']] = xp.zeros(out_channels)
        self.params[param_keys_0['conv2_w']] = xp.random.randn(out_channels, out_channels, 3, 3) * xp.sqrt(2.0 / (out_channels * 9))
        self.params[param_keys_0['conv2_b']] = xp.zeros(out_channels)
        self.params[param_keys_0['bn2_gamma']] = xp.ones(out_channels)
        self.params[param_keys_0['bn2_beta']] = xp.zeros(out_channels)

        if stride != 1 or in_channels != out_channels:
            param_keys_0.update({
                'sc_w': f'W_{stage_name}_0_shortcut', 'sc_b': f'b_{stage_name}_0_shortcut',
                'sc_bn_gamma': f'gamma_{stage_name}_0_shortcut_bn', 'sc_bn_beta': f'beta_{stage_name}_0_shortcut_bn'
            })
            self.params[param_keys_0['sc_w']] = xp.random.randn(out_channels, in_channels, 1, 1) * xp.sqrt(2.0 / in_channels)
            self.params[param_keys_0['sc_b']] = xp.zeros(out_channels)
            self.params[param_keys_0['sc_bn_gamma']] = xp.ones(out_channels)
            self.params[param_keys_0['sc_bn_beta']] = xp.zeros(out_channels)

        block0 = ResidualBlock(self.params, param_keys_0, stride)
        layers.append(block0)

        for i in range(1, num_blocks):
            param_keys_i = {
                'conv1_w': f'W_{stage_name}_{i}_conv1', 'conv1_b': f'b_{stage_name}_{i}_conv1',
                'bn1_gamma': f'gamma_{stage_name}_{i}_bn1', 'bn1_beta': f'beta_{stage_name}_{i}_bn1',
                'conv2_w': f'W_{stage_name}_{i}_conv2', 'conv2_b': f'b_{stage_name}_{i}_conv2',
                'bn2_gamma': f'gamma_{stage_name}_{i}_bn2', 'bn2_beta': f'beta_{stage_name}_{i}_bn2',
            }
            self.params[param_keys_i['conv1_w']] = xp.random.randn(out_channels, out_channels, 3, 3) * xp.sqrt(2.0 / (out_channels * 9))
            self.params[param_keys_i['conv1_b']] = xp.zeros(out_channels)
            self.params[param_keys_i['bn1_gamma']] = xp.ones(out_channels)
            self.params[param_keys_i['bn1_beta']] = xp.zeros(out_channels)
            self.params[param_keys_i['conv2_w']] = xp.random.randn(out_channels, out_channels, 3, 3) * xp.sqrt(2.0 / (out_channels * 9))
            self.params[param_keys_i['conv2_b']] = xp.zeros(out_channels)
            self.params[param_keys_i['bn2_gamma']] = xp.ones(out_channels)
            self.params[param_keys_i['bn2_beta']] = xp.zeros(out_channels)
            
            block_i = ResidualBlock(self.params, param_keys_i, stride=1)
            layers.append(block_i)
            
        return layers

    def predict(self, x, train_flg=True):
        for layer in self.layers.values():
            if isinstance(layer, list): 
                for block in layer:
                    x = block.forward(x, train_flg)
            elif isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t_fine, t_coarse, fine_to_coarse_map):
        y = self.predict(x, train_flg=False)
        y_pred_fine = xp.argmax(y, axis=1)
        
        if t_fine.ndim != 1: t_fine = xp.argmax(t_fine, axis=1)
        acc_fine = xp.sum(y_pred_fine == t_fine) / float(x.shape[0])
        
        y_pred_coarse = fine_to_coarse_map[y_pred_fine]
        if t_coarse.ndim != 1: t_coarse = xp.argmax(t_coarse, axis=1)
        acc_coarse = xp.sum(y_pred_coarse == t_coarse) / float(x.shape[0])
        
        return {'fine': acc_fine, 'coarse': acc_coarse}

    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)
        dout = 1
        dout = self.last_layer.backward(dout)
        tmp_layers = list(self.layers.values())
        tmp_layers.reverse()
        for layer_or_list in tmp_layers:
            if isinstance(layer_or_list, list):
                for block in reversed(layer_or_list):
                    dout = block.backward(dout)
            else:
                dout = layer_or_list.backward(dout)
        grads = {}
        for stage_name, num_blocks in [('', 1), ('2', 2), ('3', 2), ('4', 2), ('5', 2)]:
            if stage_name == '': # 초기 레이어
                grads['W1'] = self.layers['conv1'].dW
                grads['gamma1'] = self.layers['bn1'].dgamma
                grads['beta1'] = self.layers['bn1'].dbeta
                continue

            for i in range(num_blocks):
                block = self.layers[f'conv{stage_name}_x'][i]
                param_keys = block.param_keys

                grads[param_keys['conv1_w']] = block.conv1.dW
                grads[param_keys['conv1_b']] = block.conv1.db
                grads[param_keys['bn1_gamma']] = block.bn1.dgamma
                grads[param_keys['bn1_beta']] = block.bn1.dbeta
                
                grads[param_keys['conv2_w']] = block.conv2.dW
                grads[param_keys['conv2_b']] = block.conv2.db
                grads[param_keys['bn2_gamma']] = block.bn2.dgamma
                grads[param_keys['bn2_beta']] = block.bn2.dbeta

                if block.shortcut:
                    grads[param_keys['sc_w']] = block.shortcut.dW
                    grads[param_keys['sc_b']] = block.shortcut.db
                    grads[param_keys['sc_bn_gamma']] = block.shortcut_bn.dgamma
                    grads[param_keys['sc_bn_beta']] = block.shortcut_bn.beta

        grads['W_affine'] = self.layers['affine'].dW
        grads['b_affine'] = self.layers['affine'].db
        return grads

    def save_params(self, file_path="resnet18_params.pkl"):
        """네트워크의 모든 파라미터와 BN 통계값을 저장합니다."""
        params_to_save = {key: val for key, val in self.params.items()}

        params_to_save['bn1_running_mean'] = self.layers['bn1'].running_mean
        params_to_save['bn1_running_var'] = self.layers['bn1'].running_var

        for stage_name in ['2', '3', '4', '5']:
            for i, block in enumerate(self.layers[f'conv{stage_name}_x']):
                params_to_save[f'bn_{stage_name}_{i}_1_running_mean'] = block.bn1.running_mean
                params_to_save[f'bn_{stage_name}_{i}_1_running_var'] = block.bn1.running_var
                params_to_save[f'bn_{stage_name}_{i}_2_running_mean'] = block.bn2.running_mean
                params_to_save[f'bn_{stage_name}_{i}_2_running_var'] = block.bn2.running_var
                if block.shortcut:
                    params_to_save[f'bn_{stage_name}_{i}_s_running_mean'] = block.shortcut_bn.running_mean
                    params_to_save[f'bn_{stage_name}_{i}_s_running_var'] = block.shortcut_bn.running_var
        with open(file_path, 'wb') as f:
            pickle.dump(params_to_save, f)

    def load_params(self, file_path="resnet18_params.pkl"):
        if not os.path.exists(file_path):
            raise IOError(f"파라미터 파일을 찾을 수 없습니다: {file_path}")

        with open(file_path, 'rb') as f:
            loaded_data = pickle.load(f)

        def to_current_device(arr):
            if arr is None: return None
            if xp == np:
                return cp.asnumpy(arr) if 'cupy' in str(type(arr)) else arr
            else:
                return cp.asarray(arr) if 'numpy' in str(type(arr)) else arr

        for key, param_ref in self.params.items():
            if key in loaded_data:
                loaded_val = to_current_device(loaded_data[key])
                param_ref[:] = loaded_val

        if self.layers['bn1'].running_mean is None:
            self.layers['bn1'].running_mean = xp.zeros_like(to_current_device(loaded_data['bn1_running_mean']))
            self.layers['bn1'].running_var = xp.zeros_like(to_current_device(loaded_data['bn1_running_var']))
            
        self.layers['bn1'].running_mean[:] = to_current_device(loaded_data['bn1_running_mean'])
        self.layers['bn1'].running_var[:] = to_current_device(loaded_data['bn1_running_var'])

        for stage_name in ['2', '3', '4', '5']:
            for i, block in enumerate(self.layers[f'conv{stage_name}_x']):
                for bn_layer, mean_key, var_key in [
                    (block.bn1, f'bn_{stage_name}_{i}_1_running_mean', f'bn_{stage_name}_{i}_1_running_var'),
                    (block.bn2, f'bn_{stage_name}_{i}_2_running_mean', f'bn_{stage_name}_{i}_2_running_var'),
                    (block.shortcut_bn if block.shortcut else None, f'bn_{stage_name}_{i}_s_running_mean', f'bn_{stage_name}_{i}_s_running_var')
                ]:
                    if bn_layer is None or mean_key not in loaded_data:
                        continue
                    
                    loaded_mean = to_current_device(loaded_data[mean_key])
                    loaded_var = to_current_device(loaded_data[var_key])

                    if bn_layer.running_mean is None:
                        bn_layer.running_mean = xp.zeros_like(loaded_mean)
                        bn_layer.running_var = xp.zeros_like(loaded_var)
                    
                    bn_layer.running_mean[:] = loaded_mean
                    bn_layer.running_var[:] = loaded_var
        
        print(f"Parameters from {file_path} were successfully copied in-place into the network.")
        
def unpickle(file):
        with open(file, 'rb') as fo:
            dict_data = pickle.load(fo, encoding='latin1')
        return dict_data

def load_cifar100_data(path):
    meta = unpickle(os.path.join(path, "meta"))
    fine_label_names = meta['fine_label_names']
    coarse_label_names = meta['coarse_label_names'] 
    
    train_data = unpickle(os.path.join(path, "train"))
    X_train = train_data['data'].astype('float32').reshape(-1, 3, 32, 32) / 255.0
    y_train_fine = xp.array(train_data['fine_labels'])
    y_train_coarse = xp.array(train_data['coarse_labels']) 
    
    test_data = unpickle(os.path.join(path, "test"))
    X_test = test_data['data'].astype('float32').reshape(-1, 3, 32, 32) / 255.0
    y_test_fine = xp.array(test_data['fine_labels'])
    y_test_coarse = xp.array(test_data['coarse_labels']) 
    
    return (X_train, y_train_fine, y_train_coarse), (X_test, y_test_fine, y_test_coarse), fine_label_names, coarse_label_names

    
def evaluate_dataset(network, x_data, y_fine, y_coarse, fine_to_coarse_map, batch_size, data_name="Data Set"):
    num_data = x_data.shape[0]
    total_loss = 0
    total_acc_fine = 0
    total_acc_coarse = 0

    for i in tqdm(range(0, num_data, batch_size), desc=f"Evaluating {data_name}"):
        x_batch = x_data[i:i+batch_size]
        y_batch_fine = y_fine[i:i+batch_size]
        y_batch_coarse = y_coarse[i:i+batch_size]

        loss = network.loss(x_batch, y_batch_fine, train_flg=False)
        accs = network.accuracy(x_batch, y_batch_fine, y_batch_coarse, fine_to_coarse_map)

        current_batch_size = x_batch.shape[0]
        total_loss += loss.item() * current_batch_size
        total_acc_fine += accs['fine'].item() * current_batch_size
        total_acc_coarse += accs['coarse'].item() * current_batch_size

    final_loss = total_loss / num_data
    final_accs = {
        'fine': total_acc_fine / num_data,
        'coarse': total_acc_coarse / num_data
    }
    
    return {'loss': final_loss, 'accs': final_accs}

def main():
    # --- 데이터 로드 ---    
    (x_train, y_train_fine, y_train_coarse), \
    (x_test, y_test_fine, y_test_coarse), _, _ = load_cifar100_data(args.dataset)

    # Fine-to-Coarse 레이블 맵 생성
    y_train_fine_np = cp.asnumpy(y_train_fine) if GPU else y_train_fine
    y_train_coarse_np = cp.asnumpy(y_train_coarse) if GPU else y_train_coarse
    
    fine_to_coarse_map_np = np.zeros(100, dtype=np.int32)
    for fine_idx, coarse_idx in zip(y_train_fine_np, y_train_coarse_np):
        fine_to_coarse_map_np[fine_idx] = coarse_idx
    
    if GPU:
        x_train, y_train_fine, y_train_coarse = xp.asarray(x_train), xp.asarray(y_train_fine), xp.asarray(y_train_coarse)
        x_test, y_test_fine, y_test_coarse = xp.asarray(x_test), xp.asarray(y_test_fine), xp.asarray(y_test_coarse)
        fine_to_coarse_map = xp.asarray(fine_to_coarse_map_np)
    else:
        fine_to_coarse_map = fine_to_coarse_map_np
        
    print(f"Train: {x_train.shape[0]}개, Test: {x_test.shape[0]}개")

    # --- 모델 생성 및 가중치 로드 ---
    network = ResNet18(input_dim=(3, 32, 32), output_size=100)
    try:
        network.load_params(args.weights)
    except Exception as e:
        print(f"가중치 로드 중 오류 발생: {e}")
        sys.exit(1)

    # --- 평가 수행 ---    
    # Train 데이터셋 평가
    print("\n[Train Set 평가]")
    train_results = evaluate_dataset(network, x_train, y_train_fine, y_train_coarse, fine_to_coarse_map, args.batch_size, "Train Set")
    print(f"Loss: {train_results['loss']:.4f}")
    print(f"Fine-grained Accuracy: {train_results['accs']['fine']:.4f}")
    print(f"Super-grained Accuracy: {train_results['accs']['coarse']:.4f}")
    
    # Test 데이터셋 평가
    print("\n[Test Set 평가]")
    test_results = evaluate_dataset(network, x_test, y_test_fine, y_test_coarse, fine_to_coarse_map, args.batch_size, "Test Set")
    print(f"Loss: {test_results['loss']:.4f}")
    print(f"Fine-grained Accuracy: {test_results['accs']['fine']:.4f}")
    print(f"Super-grained Accuracy: {test_results['accs']['coarse']:.4f}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained ResNet-18 model on CIFAR-100.")
    parser.add_argument('--weights', type=str, required=True, help="Path to the pre-trained weights file (.pkl).")
    parser.add_argument('--dataset', type=str, default='./data/cifar-100-python', help="Path to the CIFAR-100 python version directory.")
    parser.add_argument('--batch-size', type=int, default=128, help="Batch size for evaluation.")
    parser.add_argument('--cpu', action='store_true', help="Force to use CPU even if GPU is available.")
    args = parser.parse_args()

    GPU = not args.cpu
    xp = np
    
    if GPU:
        import cupy as cp
        xp = cp
        print("Use GPU(CuPy)")
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    else:
        xp = np
        print("Use CPU(NumPy)")

    main()