"""
验证测试脚本的 device 适配逻辑
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
print("验证测试脚本的 device 适配逻辑")
print("="*80)

print("\n读取 test_rthd_v1_enhancements.py 并检查 device 适配...")

with open('docs0602/script/test_rthd_v1_enhancements.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("\n【检查项 1】是否在开头定义了 device")
print("-"*80)

if 'device = torch.device("cuda" if torch.cuda.is_available() else "cpu")' in content:
    print("✅ 已定义 device 变量")
    if 'print(f"Using device: {device}")' in content:
        print("✅ 打印 device 信息")
    if 'if torch.cuda.is_available():' in content[:1000]:  # 检查前1000字符
        print("✅ 根据 CUDA 可用性打印不同信息")
else:
    print("❌ 未找到 device 定义")

print("\n【检查项 2】test_tri_view_reconstruction_modes 是否使用 device")
print("-"*80)

# 查找这个函数
func_start = content.find('def test_tri_view_reconstruction_modes():')
func_end = content.find('\ndef ', func_start + 1)
func_content = content[func_start:func_end]

checks = []

# 检查输入 tensor
if 'torch.randn(B, C, H, W, device=device)' in func_content:
    print("✅ axial tensor 使用 device")
    checks.append(True)
else:
    print("❌ axial tensor 未使用 device")
    checks.append(False)

if 'torch.randn(B, C, D, W, device=device)' in func_content:
    print("✅ coronal tensor 使用 device")
    checks.append(True)
else:
    print("❌ coronal tensor 未使用 device")
    checks.append(False)

if 'torch.randn(B, C, D, H, device=device)' in func_content:
    print("✅ sagittal tensor 使用 device")
    checks.append(True)
else:
    print("❌ sagittal tensor 未使用 device")
    checks.append(False)

# 检查模块
if '.to(device)' in func_content:
    print("✅ TriViewReconstruction 模块使用 .to(device)")
    checks.append(True)
else:
    print("❌ TriViewReconstruction 模块未使用 .to(device)")
    checks.append(False)

print("\n【检查项 3】test_tri_view_vmamba_block 是否检查 CUDA")
print("-"*80)

func_start = content.find('def test_tri_view_vmamba_block():')
func_end = content.find('\ndef ', func_start + 1)
func_content = content[func_start:func_end]

if 'if not torch.cuda.is_available():' in func_content:
    print("✅ 检查 CUDA 是否可用")
    if 'return' in func_content.split('if not torch.cuda.is_available():')[1].split('\n')[0:10]:
        print("✅ 无 CUDA 时提前返回")
    if '⚠️' in func_content and '跳过' in func_content:
        print("✅ 打印跳过警告")
else:
    print("❌ 未检查 CUDA 可用性")

if 'torch.randn(B, C, D, H, W, device=device)' in func_content:
    print("✅ 输入 tensor 使用 device")
else:
    print("❌ 输入 tensor 未使用 device")

if '.to(device)' in func_content:
    print("✅ TriViewVMambaBlock 模块使用 .to(device)")
else:
    print("❌ TriViewVMambaBlock 模块未使用 .to(device)")

print("\n【检查项 4】test_rthd_block 是否检查 CUDA")
print("-"*80)

func_start = content.find('def test_rthd_block():')
func_end = content.find('\ndef ', func_start + 1)
func_content = content[func_start:func_end]

if 'if not torch.cuda.is_available():' in func_content:
    print("✅ 检查 CUDA 是否可用")
    if 'return' in func_content.split('if not torch.cuda.is_available():')[1].split('\n')[0:10]:
        print("✅ 无 CUDA 时提前返回")
else:
    print("❌ 未检查 CUDA 可用性")

if 'torch.randn(B, C, D, H, W, device=device)' in func_content:
    print("✅ 输入 tensor 使用 device")
else:
    print("❌ 输入 tensor 未使用 device")

if '.to(device)' in func_content:
    print("✅ RTHDBlock 模块使用 .to(device)")
else:
    print("❌ RTHDBlock 模块未使用 .to(device)")

print("\n【检查项 5】test_shape_compatibility 是否检查 CUDA")
print("-"*80)

func_start = content.find('def test_shape_compatibility():')
func_end = content.find('\ndef ', func_start + 1)
func_content = content[func_start:func_end]

if 'if not torch.cuda.is_available():' in func_content:
    print("✅ 检查 CUDA 是否可用")
else:
    print("❌ 未检查 CUDA 可用性")

if 'device=device' in func_content:
    print("✅ 输入 tensor 使用 device")
else:
    print("❌ 输入 tensor 未使用 device")

print("\n【检查项 6】test_backward_compatibility 是否检查 CUDA")
print("-"*80)

func_start = content.find('def test_backward_compatibility():')
func_end = content.find('\nif __name__', func_start + 1)
func_content = content[func_start:func_end]

if 'if not torch.cuda.is_available():' in func_content:
    print("✅ 检查 CUDA 是否可用")
else:
    print("❌ 未检查 CUDA 可用性")

if 'device=device' in func_content:
    print("✅ 输入 tensor 使用 device")
else:
    print("❌ 输入 tensor 未使用 device")

print("\n【检查项 7】main 函数是否区分 CPU/CUDA 测试结果")
print("-"*80)

main_start = content.find('if __name__ == "__main__":')
main_content = content[main_start:]

if 'if torch.cuda.is_available():' in main_content:
    print("✅ main 函数根据 CUDA 可用性输出不同结果")
else:
    print("❌ main 函数未区分 CPU/CUDA")

if 'skipped_tests' in main_content:
    print("✅ 跟踪跳过的测试")
else:
    print("⚠️  未跟踪跳过的测试")

if '已跳过的CUDA依赖测试' in main_content or '跳过' in main_content:
    print("✅ 明确列出跳过的测试")
else:
    print("❌ 未列出跳过的测试")

print("\n【总结】")
print("-"*80)

summary_checks = [
    ('device = torch.device' in content, "定义 device 变量"),
    ('torch.randn(B, C, H, W, device=device)' in content, "TriViewReconstruction 输入使用 device"),
    ('if not torch.cuda.is_available():' in content, "依赖 SS2D 的测试检查 CUDA"),
    ('.to(device)' in content, "模块使用 .to(device)"),
    ('⚠️' in content and '跳过' in content, "无 CUDA 时打印跳过警告"),
]

passed = sum(1 for check, _ in summary_checks if check)
total = len(summary_checks)

for check, desc in summary_checks:
    status = "✅" if check else "❌"
    print(f"{status} {desc}")

print()
if passed == total:
    print(f"🎉 所有检查通过 ({passed}/{total})")
    print()
    print("预期效果：")
    print("  【有 CUDA 环境】")
    print("    - 所有输入 tensor 和模块在 CUDA 上")
    print("    - 不会报 'Expected u.is_cuda() to be true, but got false'")
    print("    - 所有测试正常运行")
    print()
    print("  【无 CUDA 环境】")
    print("    - TriViewReconstruction 在 CPU 上测试（纯PyTorch）")
    print("    - TriViewVMambaBlock/RTHDBlock 测试被跳过（依赖CUDA）")
    print("    - 明确打印跳过原因和哪些测试被跳过")
else:
    print(f"⚠️  部分检查未通过 ({passed}/{total})")

print("="*80)
