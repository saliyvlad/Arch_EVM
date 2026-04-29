#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <iostream>

int64_t process_array_neon(const int32_t* data, size_t n) {
    int64_t sum = 0;
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    
    for (; i + 3 < n; i += 4) {
        int32x4_t vec = vld1q_s32(data + i);
        
        // ИСПРАВЛЕНО: uint32x4_t вместо int32x4_t
        uint32x4_t mask_pos = vcgtq_s32(vec, vdupq_n_s32(0));
        uint32x4_t mask_neg = vcltq_s32(vec, vdupq_n_s32(0));
        
        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);
        
        int32x4_t pos_part = vandq_s32(vec, reinterpret_cast<int32x4_t>(mask_pos));
        int32x4_t neg_part = vandq_s32(abs_val, reinterpret_cast<int32x4_t>(mask_neg));
        int32x4_t contrib = vorrq_s32(pos_part, neg_part);
        
        acc = vaddq_s32(acc, contrib);
    }
    
    // Горизонтальное сложение
    int32x2_t v1 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    int32x2_t v2 = vpadd_s32(v1, v1);
    sum = static_cast<int64_t>(vget_lane_s32(v2, 0));
    
    // Остаток
    for (; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum -= val;
    }
    
    return sum;
}
#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum += static_cast<int64_t>(std::abs(val));
    }
    return sum;
}

int64_t process_array_neon(const int32_t* data, size_t n) {
    const int32_t* aligned_data = static_cast<const int32_t*>(__builtin_assume_aligned(data, 16));
    int32x4_t acc = vdupq_n_s32(0);
    size_t i = 0;
    for (; i + 3 < n; i += 4) {
        __builtin_prefetch(aligned_data + i + 16, 0, 3);
        int32x4_t vec = vld1q_s32(aligned_data + i);
        int32x4_t zero = vdupq_n_s32(0);
        int32x4_t mask_pos = vcgtq_s32(vec, zero);
        int32x4_t mask_neg = vcltq_s32(vec, zero);
        int32x4_t sign = vshrq_n_s32(vec, 31);
        int32x4_t abs_val = veorq_s32(vec, sign);
        abs_val = vsubq_s32(abs_val, sign);
        int32x4_t pos_part = vandq_s32(vec, mask_pos);
        int32x4_t neg_part = vandq_s32(abs_val, mask_neg);
        int32x4_t contrib = vorrq_s32(pos_part, neg_part);
        acc = vaddq_s32(acc, contrib);
    }
    int32x2_t v1 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    int32x2_t v2 = vpadd_s32(v1, v1);
    int64_t sum = static_cast<int64_t>(vget_lane_s32(v2, 0));
    for (; i < n; ++i) {
        int32_t val = aligned_data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum += static_cast<int64_t>(std::abs(val));
    }
    return sum;
}

int main() {
    const size_t N = 2000000;
    int32_t* data = static_cast<int32_t*>(aligned_alloc(16, N * sizeof(int32_t)));
    if (!data) {
        std::cerr << "Memory allocation failed\n";
        return 1;
    }
    for (size_t i = 0; i < N; ++i) {
        if (i % 5 == 0) data[i] = 0;
        else if (i % 2 == 0) data[i] = static_cast<int32_t>(rand() % 100) + 1;
        else data[i] = -(static_cast<int32_t>(rand() % 100) + 1);
    }
    int64_t res_scalar = process_array_scalar(data, N);
    int64_t res_neon   = process_array_neon(data, N);
    std::cout << "Result Scalar: " << res_scalar << "\n";
    std::cout << "Result NEON  : " << res_neon   << "\n";
    std::cout << "Match        : " << (res_scalar == res_neon ? "YES" : "NO") << "\n\n";
    const int ITERATIONS = 100;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) process_array_scalar(data, N);
    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) process_array_neon(data, N);
    auto t4 = std::chrono::high_resolution_clock::now();
    double time_scalar = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double time_neon   = std::chrono::duration<double, std::milli>(t4 - t3).count();
    std::cout << "Time Scalar : " << time_scalar << " ms\n";
    std::cout << "Time NEON   : " << time_neon   << " ms\n";
    std::cout << "Speedup     : " << (time_scalar / time_neon) << "x\n";
    std::free(data);
    return 0;
}
