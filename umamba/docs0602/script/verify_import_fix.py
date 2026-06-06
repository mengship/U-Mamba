"""
验证第四次修复：SS2D 方法2导入路径
无需 torch，只检查代码逻辑
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
umamba_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, umamba_dir)
os.chdir(umamba_dir)

print("="*80)
print("验证 SS2D 方法2导入路径修复")
print("="*80)

print("\n读取 rthd_modules.py 并检查导入逻辑...")

with open('nnunetv2/nets/rthd_modules.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 查找 SS2D 导入部分
import_start = content.find('# 导入SS2D（2D VMamba核心模块）')
import_end = content.find('# 根据 share_weights 决定实例化方式', import_start)
import_section = content[import_start:import_end]

print("\n【检查项 1】方法2是否正确计算项目根目录")
print("-"*80)

if 'project_root = os.path.dirname(umamba_dir)' in import_section:
    print("✅ 方法2正确计算 project_root（umamba的父目录）")
    print("   project_root = os.path.dirname(umamba_dir)")
else:
    print("❌ 方法2未正确计算 project_root")

print("\n【检查项 2】方法2是否将 project_root 加入 sys.path")
print("-"*80)

if 'if os.path.exists(project_root) and project_root not in sys.path:' in import_section:
    print("✅ 方法2检查并添加 project_root 到 sys.path")
    if 'sys.path.insert(0, project_root)' in import_section:
        print("✅ 使用 sys.path.insert(0, project_root)")
    else:
        print("❌ 未使用 sys.path.insert")
else:
    print("❌ 方法2未将 project_root 加入 sys.path")

print("\n【检查项 3】方法2是否使用包导入方式")
print("-"*80)

if 'from umamba.instructions.vmamba import SS2D' in import_section:
    print("✅ 方法2使用包导入：from umamba.instructions.vmamba import SS2D")
else:
    print("❌ 方法2未使用包导入方式")

print("\n【检查项 4】日志优化")
print("-"*80)

# 检查方法1失败日志
if '⚠️  Method 1' in import_section or 'Method 1 (direct import) failed' in import_section:
    print("✅ 方法1失败时使用 warning (⚠️) 而非 error")
else:
    print("⚠️  方法1失败日志未优化为 warning")

# 检查方法2失败日志
if '⚠️  Method 2' in import_section or 'Method 2 (package import) failed' in import_section:
    print("✅ 方法2失败时使用 warning (⚠️) 而非 error")
else:
    print("⚠️  方法2失败日志未优化为 warning")

# 检查总错误日志逻辑
if 'if SS2D is None:' in import_section:
    none_check_pos = import_section.find('if SS2D is None:', import_section.find('# 方法3'))
    if none_check_pos > 0:
        print("✅ 只有两个方法都失败时才打印总错误（if SS2D is None 在方法3）")
    else:
        print("⚠️  总错误逻辑可能有问题")

print("\n【检查项 5】预期的 sys.path 层级")
print("-"*80)

print("修复后，方法2会将以下路径加入 sys.path：")
print("  /home/wang/U-Mamba  （项目根目录，umamba的父目录）")
print()
print("这样 Python 就能正确解析：")
print("  from umamba.instructions.vmamba import SS2D")
print("  -> /home/wang/U-Mamba/umamba/instructions/vmamba.py")
print()
print("修复前的问题：")
print("  sys.path 只有 /home/wang/U-Mamba/umamba")
print("  Python 会去找 /home/wang/U-Mamba/umamba/umamba （错误！）")

print("\n【检查项 6】路径计算流程")
print("-"*80)

# 查找路径计算部分
if 'current_dir = os.path.dirname(os.path.abspath(__file__))' in import_section:
    print("✅ Step 1: current_dir = os.path.dirname(os.path.abspath(__file__))")
    print("   示例: /home/wang/U-Mamba/umamba/nnunetv2/nets")

if 'umamba_dir = os.path.dirname(os.path.dirname(current_dir))' in import_section:
    print("✅ Step 2: umamba_dir = os.path.dirname(os.path.dirname(current_dir))")
    print("   示例: /home/wang/U-Mamba/umamba")

if 'project_root = os.path.dirname(umamba_dir)' in import_section:
    print("✅ Step 3: project_root = os.path.dirname(umamba_dir)")
    print("   示例: /home/wang/U-Mamba")
    print("   （这是方法2新增的关键步骤）")

print("\n【总结】")
print("-"*80)

checks = [
    ('project_root = os.path.dirname(umamba_dir)' in import_section, "计算项目根目录"),
    ('if os.path.exists(project_root) and project_root not in sys.path:' in import_section, "添加项目根目录到 sys.path"),
    ('from umamba.instructions.vmamba import SS2D' in import_section, "使用包导入"),
    ('⚠️  Method 1' in import_section or 'Method 1 (direct import) failed' in import_section, "方法1失败日志优化"),
    ('⚠️  Method 2' in import_section or 'Method 2 (package import) failed' in import_section, "方法2失败日志优化"),
]

passed = sum(1 for check, _ in checks if check)
total = len(checks)

for check, desc in checks:
    status = "✅" if check else "❌"
    print(f"{status} {desc}")

print()
if passed == total:
    print(f"🎉 所有检查通过 ({passed}/{total})")
    print()
    print("预期效果：")
    print("  - 方法2不再报 'No module named umamba'")
    print("  - sys.path 中会看到 /home/wang/U-Mamba")
    print("  - 如果方法1失败但方法2成功，不会打印总错误")
else:
    print(f"⚠️  部分检查未通过 ({passed}/{total})")

print("="*80)
