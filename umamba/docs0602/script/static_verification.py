"""
静态验证脚本 - 验证代码修复（无需torch）
检查三个问题的修复是否正确
"""

import sys
import os

# 添加路径：从脚本位置向上找到 umamba 目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir: .../umamba/docs0602/script
# umamba_dir: .../umamba
umamba_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, umamba_dir)

# 切换工作目录到 umamba_dir，使相对路径正确
os.chdir(umamba_dir)

print("="*80)
print("静态验证：检查三个问题的修复")
print("="*80)

# 问题1：特征幅值问题 - 检查_apply_cross_view_interaction是否使用残差式门控
print("\n【问题 1】检查：cross-view interaction 是否使用残差式门控")
print("-"*80)

with open('nnunetv2/nets/rthd_modules.py', 'r', encoding='utf-8') as f:
    content = f.read()

    # 检查是否改为tanh
    if 'torch.tanh(gates_3d' in content:
        print("✅ 门控激活函数已改为 tanh（范围[-1, 1]）")
    else:
        print("❌ 未找到 tanh 激活函数")

    # 检查是否使用残差式门控
    if 'axial + axial * gate_axial' in content or 'axial * (1 + gate_axial)' in content:
        print("✅ 使用残差式门控：axial + axial * gate_axial")
    else:
        print("❌ 未找到残差式门控")

    # 确认没有直接相乘
    if 'axial_refined = axial * gate_axial' in content and 'axial + axial * gate_axial' not in content:
        print("❌ 仍在使用直接相乘（会压低特征幅值）")
    else:
        print("✅ 已移除直接相乘方式")

# 问题2：参数接口安全 - 检查是否有参数校验
print("\n【问题 2】检查：interaction 参数是否有严格校验")
print("-"*80)

with open('nnunetv2/nets/rthd_modules.py', 'r', encoding='utf-8') as f:
    content = f.read()

    # 检查是否有interaction_mode校验
    if "interaction_mode != 'post'" in content and 'raise ValueError' in content:
        print("✅ 添加了 interaction_mode 参数校验")
    else:
        print("❌ 未找到 interaction_mode 参数校验")

    # 检查是否有interaction_type校验
    if "interaction_type != 'gate'" in content and 'raise ValueError' in content:
        print("✅ 添加了 interaction_type 参数校验")
    else:
        print("❌ 未找到 interaction_type 参数校验")

    # 检查是否在构造时就校验
    # 查找__init__中cross_view_interaction的位置
    init_start = content.find('def __init__(', content.find('class TriViewVMambaBlock'))
    init_end = content.find('def forward(', init_start)
    init_section = content[init_start:init_end]

    if 'raise ValueError' in init_section and 'interaction_mode' in init_section:
        print("✅ 参数校验在 __init__ 阶段执行（快速失败）")
    else:
        print("⚠️  参数校验可能不在 __init__ 阶段")

# 问题3：测试脚本fallback分支 - 检查是否处理格式不匹配
print("\n【问题 3】检查：fallback 分支是否处理 NCHW/NHWC 格式兼容")
print("-"*80)

with open('nnunetv2/nets/rthd_modules.py', 'r', encoding='utf-8') as f:
    content = f.read()

    # 检查是否记录using_real_ss2d
    if 'self.using_real_ss2d' in content:
        print("✅ 添加了 using_real_ss2d 标记")
    else:
        print("❌ 未找到 using_real_ss2d 标记")

    # 检查是否在_process_view中根据标记选择格式
    if 'if self.using_real_ss2d:' in content:
        print("✅ _process_view 根据 using_real_ss2d 选择格式")
    else:
        print("❌ _process_view 未根据标记选择格式")

    # 检查placeholder是否统一为channels-last
    # 查找placeholder定义
    if 'nn.LayerNorm(dim)' in content and 'nn.Linear(dim, dim)' in content:
        print("✅ Placeholder 使用 LayerNorm + Linear（channels-last兼容）")
    else:
        print("⚠️  Placeholder 实现可能需要检查")

    # 确认没有Conv2d的placeholder
    placeholder_section = content[content.find('else:'):content.find('self.vmamba_2d = None')]
    if 'nn.Conv2d' in placeholder_section:
        print("⚠️  仍有 Conv2d placeholder（可能导致格式不匹配）")
    else:
        print("✅ 已移除 Conv2d placeholder")

# 兼容性检查
print("\n【兼容性】检查：是否保持向后兼容")
print("-"*80)

with open('nnunetv2/nets/rthd_modules.py', 'r', encoding='utf-8') as f:
    content = f.read()

    # 检查原有参数是否保留
    required_params = ['view_mode', 'share_weights', 'scan_mode', 'use_local_window']
    all_present = all(param in content for param in required_params)

    if all_present:
        print("✅ 所有原有参数保持不变")
    else:
        print("❌ 部分原有参数缺失")

    # 检查原有reconstruction_mode是否保留
    if "'broadcast'" in content and "'weighted'" in content and "'gated'" in content:
        print("✅ 三种 reconstruction_mode 保持支持")
    else:
        print("❌ reconstruction_mode 支持不完整")

print("\n" + "="*80)
print("静态验证完成")
print("="*80)

print("\n总结：")
print("  - 问题 1（特征幅值）：检查残差式门控")
print("  - 问题 2（参数校验）：检查快速失败机制")
print("  - 问题 3（格式兼容）：检查 NCHW/NHWC 处理")
print("  - 兼容性：检查原有功能保持")

print("\n注意：由于无torch环境，无法运行实际前向测试")
print("建议在有torch环境时运行完整测试脚本验证")
print("="*80)
