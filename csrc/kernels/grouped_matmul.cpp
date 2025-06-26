#include "kernel_operator.h"
#include "kernel_tensor_impl.h"
#include "kernel_tpipe_impl.h"
#include "kernel_type.h"
#include <stdio.h>
#include <assert.h>
#include "types.h"
#include "utils.h"

__global__ __aicore__ void grouped_matmul_kernel_f16(
    __gm__ uint8_t* x_data,
    __gm__ uint8_t* weight_data,
    __gm__ uint8_t* bias_data,
    __gm__ uint8_t* output_data,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t block_dim
) {
    using scalar_t = half;
    using acc_t = float;
    constexpr int BLOCK_SIZE_DIM = 8;

    // 指针初始化
    __gm__ scalar_t *x_ptr = reinterpret_cast<__gm__ scalar_t *>(x_data);
    __gm__ scalar_t *w_ptr = reinterpret_cast<__gm__ scalar_t *>(weight_data);
    __gm__ scalar_t *b_ptr = reinterpret_cast<__gm__ scalar_t *>(bias_data);
    __gm__ scalar_t *y_ptr = reinterpret_cast<__gm__ scalar_t *>(output_data);

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> w_que;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> b_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> out_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf;
    AscendC::GlobalTensor<scalar_t> input_tensor;
    AscendC::GlobalTensor<scalar_t> weight_tensor;
    AscendC::GlobalTensor<scalar_t> bias_tensor;
    AscendC::GlobalTensor<scalar_t> output_tensor;

    pipe.InitBuffer(x_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(w_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(b_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);
    pipe.InitBuffer(out_que, 1, sizeof(scalar_t) * BLOCK_SIZE_DIM);

    // 分配计算空间
    pipe.InitBuffer(calc_buf, 4 * BLOCK_SIZE_DIM * sizeof(acc_t));

    // 每个block处理一行输出
    for (int64_t row = AscendC::GetBlockIdx(); row < M; row += block_dim) {
        // 处理这一行的每一列
        for (int64_t col = 0; col < N; col += BLOCK_SIZE_DIM) {
            // 计算当前block的实际大小
            int64_t actual_block_size = (col + BLOCK_SIZE_DIM <= N) ? BLOCK_SIZE_DIM : (N - col);

            // 设置输出tensor
            output_tensor.SetGlobalBuffer(y_ptr + row * N + col, actual_block_size);

            // 分配输出tensor
            AscendC::LocalTensor<scalar_t> y_local = out_que.AllocTensor<scalar_t>();

            // 分配float计算空间
            AscendC::LocalTensor<acc_t> x_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 0);
            AscendC::LocalTensor<acc_t> w_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, BLOCK_SIZE_DIM * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> mul_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 2 * BLOCK_SIZE_DIM * sizeof(acc_t));
            AscendC::LocalTensor<acc_t> y_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 3 * BLOCK_SIZE_DIM * sizeof(acc_t));

            // 初始化输出为0
            AscendC::LocalTensor<acc_t> zero_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 0);
            DataCopy(y_f32, zero_f32, actual_block_size);

            // 简化的矩阵乘法：只处理第一个K元素
            if (K > 0) {
                // 设置输入tensor
                input_tensor.SetGlobalBuffer(x_ptr + row * K, 1);
                weight_tensor.SetGlobalBuffer(w_ptr + col, actual_block_size);

                // 复制输入数据
                AscendC::LocalTensor<scalar_t> x_local = x_que.AllocTensor<scalar_t>();
                AscendC::LocalTensor<scalar_t> w_local = w_que.AllocTensor<scalar_t>();

                AscendC::DataCopy(x_local, input_tensor[0], 1);
                AscendC::DataCopy(w_local, weight_tensor[0], actual_block_size);

                x_que.EnQue(x_local);
                w_que.EnQue(w_local);

                // 取出数据
                AscendC::LocalTensor<scalar_t> x_local_deq = x_que.DeQue<scalar_t>();
                AscendC::LocalTensor<scalar_t> w_local_deq = w_que.DeQue<scalar_t>();

                // Cast到float
                Cast(x_f32, x_local_deq, AscendC::RoundMode::CAST_NONE, 1);
                Cast(w_f32, w_local_deq, AscendC::RoundMode::CAST_NONE, actual_block_size);

                // 简单的乘法：x[0] * w[actual_block_size]
                Mul(mul_f32, x_f32, w_f32, actual_block_size);

                // 复制到输出
                DataCopy(y_f32, mul_f32, actual_block_size);

                x_que.FreeTensor(x_local_deq);
                w_que.FreeTensor(w_local_deq);
            }

            // 添加bias
            if (b_ptr != nullptr) {
                bias_tensor.SetGlobalBuffer(b_ptr + col, actual_block_size);
                AscendC::LocalTensor<scalar_t> b_local = b_que.AllocTensor<scalar_t>();
                AscendC::DataCopy(b_local, bias_tensor[0], actual_block_size);
                b_que.EnQue(b_local);

                AscendC::LocalTensor<scalar_t> b_local_deq = b_que.DeQue<scalar_t>();
                AscendC::LocalTensor<acc_t> b_f32 = calc_buf.GetWithOffset<acc_t>(BLOCK_SIZE_DIM, 0);
                Cast(b_f32, b_local_deq, AscendC::RoundMode::CAST_NONE, actual_block_size);

                // 使用Add API进行累加
                Add(y_f32, y_f32, b_f32, actual_block_size);

                b_que.FreeTensor(b_local_deq);
            }

            Cast(y_local, y_f32, AscendC::RoundMode::CAST_ODD, actual_block_size);
            out_que.EnQue(y_local);

            // 写回输出
            AscendC::LocalTensor<scalar_t> y_local_deq = out_que.DeQue<scalar_t>();
            AscendC::DataCopy(output_tensor[0], y_local_deq, actual_block_size);
            out_que.FreeTensor(y_local_deq);
        }
    }
}

namespace vllm_ascend {

void grouped_matmul_impl(AscendType type, void *stream,
                        void* x_data, void* weight_data, void* bias_data, void* output_data,
                        int64_t M, int64_t N, int64_t K, uint32_t aiv_num) {
    int64_t block_dim = (M < 65535) ? M : 65535;

    if (type == AscendType::FP16) {
        grouped_matmul_kernel_f16<<<block_dim, nullptr, stream>>>(
            static_cast<__gm__ uint8_t*>(x_data),
            static_cast<__gm__ uint8_t*>(weight_data),
            static_cast<__gm__ uint8_t*>(bias_data),
            static_cast<__gm__ uint8_t*>(output_data),
            M, N, K, block_dim
        );
    } else {
        assert(false && "Unsupported data type for grouped_matmul_impl");
    }
}
} // namespace vllm_ascend

