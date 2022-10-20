
#include <diopi/functions.h>
#include <torch/nn.h>
#include <torch/optim.h>
#include <iostream>
#include <math.h>

#include "helper.hpp"
#include "vision_kernel.h"

#define FLT_MIN		__FLT_MIN__

extern "C" {

diopiError_t diopiRelu(diopiContextHandle_t ctx,
        diopiTensorHandle_t out, const diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncRet(ctx, at::relu, out, atInput);
    return diopiSuccess;
}

diopiError_t diopiReluInp(diopiContextHandle_t ctx, diopiTensorHandle_t input) {
    at::Tensor atInput = impl::aten::buildATen(input);
    impl::aten::invokeATenFuncInp(ctx, at::relu_, atInput);
    return diopiSuccess;
}

}  // extern "C"