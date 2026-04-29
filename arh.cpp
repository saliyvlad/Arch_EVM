#include <arm_neon.h>
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>

// ==========================================
// 1. СКАЛЯРНАЯ ВЕРСИЯ (эталон)
// ==========================================
int64_t process_array_scalar(const int32_t* data, size_t n) {
    int64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        int32_t val = data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum += static_cast<int64_t>(std::abs(val));
        // val == 0 пропускается
    }
    return sum;
}

// ==========================================
// 2. ВЕКТОРНАЯ ВЕРСИЯ (ARM NEON)
// ==========================================
int64_t process_array_neon(const int32_t* data, size_t n) {
    // Сообщаем компилятору о выравнивании памяти (отключает штраф за невыровненный доступ)
    const int32_t* aligned_data = static_cast<const int32_t*>(__builtin_assume_aligned(data, 16));

    int32x4_t acc = vdupq_n_s32(0); // Векторный аккумулятор
    size_t i = 0;

    // Основной цикл: обработка по 4 элемента за итерацию
    for (; i + 3 < n; i += 4) {
        // Продвинутая оптимизация: предзагрузка следующей строки кэша
        __builtin_prefetch(aligned_data + i + 16, 0, 3);

        // 1. Множественная загрузка (аналог LDM для NEON)
        int32x4_t vec = vld1q_s32(aligned_data + i);

        // 3. Условное выполнение без ветвлений (битовые маски)
        int32x4_t zero      = vdupq_n_s32(0);
        int32x4_t mask_pos  = vcgtq_s32(vec, zero); // 0xFFFFFFFF если > 0, иначе 0x00000000
        int32x4_t mask_neg  = vcltq_s32(vec, zero); // 0xFFFFFFFF если < 0, иначе 0x00000000

        // 2. Безветвевое вычисление модуля через баррельный шифтер
        int32x4_t sign      = vshrq_n_s32(vec, 31);      // Арифметический сдвиг: -1 для отрицательных, 0 для остальных
        int32x4_t abs_val   = veorq_s32(vec, sign);      // Инверсия битов для отрицательных чисел
        abs_val             = vsubq_s32(abs_val, sign);  // Вычитание маски: ~x - (-1) == |x|

        // Маскирование: положительные оставляем, отрицательные заменяем на модуль, нули обнуляются
        int32x4_t pos_part  = vandq_s32(vec, mask_pos);
        int32x4_t neg_part  = vandq_s32(abs_val, mask_neg);
        int32x4_t contrib   = vorrq_s32(pos_part, neg_part); // Объединение результатов

        // Накопление в векторном регистре
        acc = vaddq_s32(acc, contrib);
    }

    // 5. Горизонтальное сложение (совместимо с ARMv7 NEON)
    int32x2_t v1 = vadd_s32(vget_low_s32(acc), vget_high_s32(acc));
    int32x2_t v2 = vpadd_s32(v1, v1);
    int64_t sum  = static_cast<int64_t>(vget_lane_s32(v2, 0));

    // Обработка остатка (0-3 элемента)
    for (; i < n; ++i) {
        int32_t val = aligned_data[i];
        if (val > 0) sum += val;
        else if (val < 0) sum += static_cast<int64_t>(std::abs(val));
    }

    return sum;
}

// ==========================================
// ТЕСТ И БЕНЧМАРК
// ==========================================
int main() {
    const size_t N = 2000000; // Размер >= 1000 для корректного замера ускорения
    
    // Выделение памяти, выровненной на 16 байт
    int32_t* data = static_cast<int32_t*>(aligned_alloc(16, N * sizeof(int32_t)));
    if (!data) {
        std::cerr << "Ошибка выделения выровненной памяти!\n";
        return 1;
    }

    // Заполнение: ~20% нулей, остальные случайные положительные/отрицательные
    for (size_t i = 0; i < N; ++i) {
        if (i % 5 == 0) data[i] = 0;
        else if (i % 2 == 0) data[i] = static_cast<int32_t>(rand() % 100) + 1;
        else data[i] = -(static_cast<int32_t>(rand() % 100) + 1);
    }

    // Проверка корректности
    int64_t res_scalar = process_array_scalar(data, N);
    int64_t res_neon   = process_array_neon(data, N);

    std::cout << "Результат (Scalar): " << res_scalar << "\n";
    std::cout << "Результат (NEON)  : " << res_neon   << "\n";
    std::cout << "Совпадение        : " << (res_scalar == res_neon ? "✅ ДА" : "❌ НЕТ") << "\n\n";

    // Бенчмарк производительности
    const int ITERATIONS = 100;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) process_array_scalar(data, N);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto t3 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) process_array_neon(data, N);
    auto t4 = std::chrono::high_resolution_clock::now();

    double time_scalar = std::chrono::duration<double, std::milli>(t2 - t1).count();
    double time_neon   = std::chrono::duration<double, std::milli>(t4 - t3).count();

    std::cout << "Время Scalar : " << time_scalar << " мс\n";
    std::cout << "Время NEON   : " << time_neon   << " мс\n";
    std::cout << "Ускорение    : " << (time_scalar / time_neon) << "x\n";

    std::free(data);
    return 0;
}