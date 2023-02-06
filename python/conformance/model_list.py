model_list = ['resnet50', 'vgg16', 'resnet101', 'seresnet50']

model_op_list = {
    'resnet50' : ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    'resnet101' : ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    'seresnet50' : ['sgd', 'randperm', 'conv2d', 'add', 'batch_norm', 'relu', 'adaptive_avg_pool2d', 'sigmoid', 'linear', 'cross_entropy', 'sum', 'mean', 'mul', 'div'],
    'vgg16' : ['sgd', 'random', 'randperm', 'conv2d', 'relu', 'batch_norm', 'max_pool2d', 'linear', 'addmm', 'dropout', 'cross_entropy', 'log_softmax', 'nll_loss', 'sum', 'mean', 'add',],
}
