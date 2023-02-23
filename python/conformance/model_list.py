model_list = ['resnet50', 'vgg16', 'resnet101', 'seresnet50', 'densenet', 'mobilenet_v2',
              'efficientnet', 'shufflenet_v2', 'repvgg', 'swin_transformer', 'vit', 'inceptionv3'
              'retinanet', 'faster_rcnn_r50', 'ssd300', 'yolov3',
              'unet', 'upernet', 'pspnet', 'fcn', 'deeplabv3', 'deeplabv3plus']

model_op_list = {
    'resnet50': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    'resnet101': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    'seresnet50': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'sigmoid', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    # densenet is too large to train on 1 GPU
    'densenet': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'cat', 'avg_pool2d'],

    'mobilenet_v2': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'hardtanh', 'sum', 'mean', 'mul', 'div'],
    'efficientnet': ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'linspace', 'pad', 'sigmoid', 'sum', 'mean', 'mul', 'div'],
    'vgg16': ['sgd', 'randperm', 'conv2d', 'relu', 'batch_norm', 'max_pool2d', 'linear', 'dropout', 'cross_entropy', 'sum', 'mean', 'add', 'mul', 'div'],
    'shufflenet_v2': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'cat', 'transpose', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'sgd', 'mul', 'div'],
    'repvgg': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'sgd', 'mul', 'div'],
    'swin_transformer': ['linspace', 'arange', 'randperm', 'one_hot', 'mul', 'add', 'conv2d', 'transpose', 'layer_norm', 'dropout',
                         'pad', 'permute', 'linear', 'matmul', 'softmax', 'gelu', 'roll', 'sub', 'ne', 'masked_fill', 'eq', 'rand', 'div', 'floor', 'unfold', 'linear', 'adaptive_avg_pool2d', 'neg',
                         'log_softmax', 'sum', 'mean', 'norm', 'stack', 'reciprocal', 'clamp', 'adamw', 'addcmul', 'sqrt', 'addcdiv'],
    'vit': ['randperm', 'one_hot', 'mul', 'add', 'conv2d', 'transpose', 'expand', 'cat', 'dropout', 'layer_norm', 'linear',
            'permute', 'matmul', 'gelu', 'tanh', 'neg', 'log_softmax', 'sum', 'div', 'mean', 'norm', 'stack', 'reciprocal', 'clamp', 'adamw', 'addcmul', 'sqrt', 'addcdiv'],
    'inceptionv3': [],
    'ssd300': ['maximum', 'randperm', 'conv2d', 'relu', 'max_pool2d', 'pow', 'sum', 'sqrt', 'add',  'expand', 'mul', 'div', 'arange', 'stack', 'logical_and', 'cat', 'maximum', 'minimum',
                      'sub', 'max', 'min', 'clamp', 'ge', 'lt', 'gt', 'index_select', 'log', 'permute', 'cross_entropy', 'topk', 'abs', 'where', 'mean', 'eq', 'sgd', 'nonzero', 'sort', 'exp'],
    'retinanet': ['randperm',  'conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add',  'interpolate', 'orange', 'mul', 'stack', 'logical_and', 'expand', 'cat', 'any', 'sub', 'maximum',
                  'minimum', 'clamp', 'div', 'max', 'ge', 'lt', 'eq', 'gt', 'nonzero', 'unique', 'log', 'permute', 'sum', 'abs', 'mean', 'sgd', 'sigmoid', 'sort'],
    'faster_rcnn_r50': ['randperm',  'conv2d', 'batch_norm', 'relu', 'max_pool2d', 'add',  'interpolate', 'orange', 'mul', 'stack', 'logical_and', 'expand', 'cat', 'any', 'sub', 'maximum',
                        'minimum', 'clamp', 'div', 'max', 'ge', 'lt', 'eq', 'gt', 'nonzero', 'unique', 'log', 'permute', 'ne', 'binary_cross_entropy_with_logits',
                        'sum', 'abs',  'sigmoid', 'sort', 'exp', 'all', 'sort', 'log2', 'floor', 'linear',  'cross_entropy', 'topk',  'transpose', 'mean', 'sgd',  'split', 'softmax'],
    'yolov3': ['randperm',  'conv2d', 'batch_norm', 'leaky_relu', 'add', 'interpolate',  'cat', 'arange',  'mul',  'stack', 'div', 'floor', 'expand', 'sub',  'maximum',  'minimum',
               'clamp', 'max', 'ge', 'le',  'logical_and', 'bitwise_not', 'gt',  'eq',  'nonzero',  'unique',  'log', 'one_hot', 'permute', 'ne', 'binary_cross_entropy_with_logits', 'sum',
               'mse_loss',  'mean',  'norm', 'reciprocal', 'sgd', 'sigmoid', 'exp'],
    'unet': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'upernet': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'pspnet': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'fcn': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'cat', 'interpolate', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'deeplabv3': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],
    'deeplabv3plus': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'adaptive_avg_pool2d', 'interpolate', 'cat', 'cross_entropy', 'mean', 'mul', 'topk', 'transpose', 'expand', 'eq', 'ne', 'sum', 'div', 'sgd'],

}
