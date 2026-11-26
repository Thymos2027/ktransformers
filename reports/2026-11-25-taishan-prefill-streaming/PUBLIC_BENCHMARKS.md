# 公开prefill参考及解释边界

核对日期：2026-11-25，北京时间。

[KT Ascend教程](https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md#measured-results)报告单颗A3 Ascend910_9362、61.3GB HBM、40核/1 NUMA、CPU32线程/1池、32常驻专家、单请求、graph decode。801/3944/7823输入的prefill为15.6/16.0/16.5秒；启用streaming、depool、GGUF dedup、动态热专家和侧流重叠。

[CANN方案文档](https://github.com/hicann/cann-recipes-infer/blob/master/docs/integration/sglang/dsv4-flash-single-npu-moe-offload/dsv4_flash_single_card_design.md)另报4K hybrid约137秒、streaming约14秒，并解释长prefill将每层全部专家搬到NPU计算。该组是另一文档的参考，不能与KT表格合并为同一批原始测试。

[公开测量脚本](https://github.com/hicann/cann-recipes-infer/blob/master/integration/sglang/dsv4-flash-single-npu-moe-offload/scripts/tools/decode_throughput_test.sh)把首次正文增长的等待称为prefill；本机同时报告首输出token与scheduler墙钟prefill，两者口径不完全相同。本机双卡已有1024/4096/7168输入prefill为12.340/47.703/82.964秒，KT_PREFILL_STREAM=0，不能把跨硬件差距直接当作A3算力倍数。

streaming先付每请求权重搬运成本，再用NPU批量算专家，长输入可摊薄固定成本；短输入可能退化。CPU、NUMA、NPU代际、链路、内核版本、提示和预热均不同，公开16秒不作为本机承诺。
