model_list = ['resnet50', 'vgg16', 'resnet101', 'seresnet50', 'densenet', 'mobilenet_v2',
              'efficientnet', 'shufflenet_v2', 'repvgg']

model_op_list = {
    'resnet50' : ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    'resnet101' : ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    'seresnet50' : ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'sigmoid', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    # densenet is too large to train on 1 GPU
    'densenet' : ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'cat', 'avg_pool2d'],

    'mobilenet_v2' : ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'hardtanh', 'sum', 'mean', 'mul', 'div'],
    'efficientnet' : ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'linspace', 'pad', 'sigmoid', 'sum', 'mean', 'mul', 'div'],
    'vgg16' : ['sgd', 'randperm', 'conv2d', 'relu', 'batch_norm', 'max_pool2d', 'linear', 'dropout', 'cross_entropy', 'sum', 'mean', 'add', 'mul', 'div'],
    'shufflenet_v2': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'max_pool2d', 'cat', 'transpose', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'sgd', 'mul', 'div'],
    'repvgg': ['randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'sgd', 'mul', 'div']
}
