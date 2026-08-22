#!/usr/bin/env bash
# 永续载体上跑舰队引擎的标准 payload(任何槽通用)。
#
# 为什么不直接 clear 回默认:载体默认 payload 就是 corr_wave_fleet.sh,但它是被
# 载体以 SLOT=<本槽> 显式导出后调用的,而引擎对"显式 SLOT + rank!=0"的组合有一条
# 单槽重复保护会让 pod 空转(2026-08-23 烧掉 24 卡 × 19.5 小时的教训)。载体给每个
# pod 分的是各自的槽,不是副本,所以这里声明 CORR_SLOT_OWNED=1 豁免那条守卫;
# 真正的互斥仍由引擎的槽锁 + 载体心跳认领把守。
#
# 用法:task.sh set/swap <槽> $ROOT/deploy/dlc/slot_resume.sh
# lane 清单照旧走覆盖文件 $D/corr_wave/slot<k>_s<seed>_lanes。
set -u
export CORR_SLOT_OWNED=1
exec bash "${ROOT:-/mgfs/shared/Group_GY/changhao/SimOPD-exp}/deploy/dlc/corr_wave_fleet.sh"
