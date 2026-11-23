// 由 transnoise.deploy.pack 生成，不要手改。模型形状全部是编译期常量；weights.bin 的头里带同一组常量，加载时逐项校验。
// 来源：检查点 step 96000
#pragma once
#include <cstdint>

namespace tn {
inline constexpr int SR = 48000, FFT = 960, HOP = 480, BINS = FFT / 2 + 1;
inline constexpr int NB_ERB = 32, NB_DF = 192, ORDER = 7;
inline constexpr int D = 168, LAYERS = 3, HEADS = 4, DH = D / HEADS, FF = D * 2;
inline constexpr int WINDOW = 100, LA = 1;
inline constexpr float ERB_DB_OFFSET = 60.0f, ERB_DB_SCALE = 40.0f, LN_EPS = 0.0010000000474974513f;
inline constexpr int ERB_WIDTHS[NB_ERB] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 7, 7, 8, 10, 12, 13, 15, 18, 20, 24, 28, 31, 37, 42, 50, 56, 67};
inline constexpr int32_t CONSTS[] = {48000, 960, 480, 32, 192, 168, 3, 4, 2, 100, 1, 7};  // 顺序：sr fft hop nb_erb nb_df d_model n_layers n_heads ff_mult window lookahead df_order
}  // namespace tn
