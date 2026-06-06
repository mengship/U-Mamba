"""
简单导入测试 - 验证测试脚本的导入路径是否正确
无需 torch，只测试模块导入
"""

import sys
import os

# 添加路径：从脚本位置向上找到 umamba 目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir: .../umamba/docs0602/script
# umamba_dir: .../umamba
umamba_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, umamba_dir)

print("="*80)
print("导入路径测试")
print("="*80)

print(f"\n当前脚本路径: {current_dir}")
print(f"umamba 目录路径: {umamba_dir}")
print(f"sys.path[0]: {sys.path[0]}")

# 测试导入
print("\n尝试导入 rthd_modules...")
try:
    from nnunetv2.nets.rthd_modules import (
        TriViewReconstruction,
        TriViewVMambaBlock,
        RTHDBlock
    )
    print("✅ 成功导入 TriViewReconstruction")
    print("✅ 成功导入 TriViewVMambaBlock")
    print("✅ 成功导入 RTHDBlock")

    # 检查类是否可实例化（不需要实际创建，只检查类存在）
    print("\n检查类定义...")
    print(f"✅ TriViewReconstruction 是类: {type(TriViewReconstruction).__name__}")
    print(f"✅ TriViewVMambaBlock 是类: {type(TriViewVMambaBlock).__name__}")
    print(f"✅ RTHDBlock 是类: {type(RTHDBlock).__name__}")

    print("\n" + "="*80)
    print("✅ 导入路径配置正确！")
    print("="*80)

except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    print("\n可能的原因:")
    print("  1. sys.path 设置不正确")
    print("  2. nnunetv2 模块不在正确位置")
    print("  3. rthd_modules.py 有语法错误")
    print("\n" + "="*80)
    sys.exit(1)
except Exception as e:
    print(f"\n⚠️  导入成功但出现其他错误: {e}")
    print("这可能是因为缺少依赖（如 torch）")
    print("但导入路径本身是正确的")
    print("\n" + "="*80)
    sys.exit(0)
