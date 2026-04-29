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
