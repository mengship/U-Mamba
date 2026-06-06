"""
验证 fallback placeholder 格式处理逻辑
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
print("验证 fallback placeholder 格式处理逻辑")
print("="*80)

print("\n读取 rthd_modules.py 并检查 _process_view 方法...")

with open('nnunetv2/nets/rthd_modules.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 检查 _process_view 方法
process_view_start = content.find('def _process_view(self')
process_view_end = content.find('\n\nclass ', process_view_start)
process_view_code = content[process_view_start:process_view_end]

print("\n【检查项 1】全局平铺版 (use_local_window=False)")
print("-"*80)

# 检查是否统一做了 permute
if 'if not self.use_local_window:' in process_view_code:
    # 找到这个分支
    no_window_start = process_view_code.find('if not self.use_local_window:')
    no_window_end = process_view_code.find('else:', no_window_start)
    no_window_branch = process_view_code[no_window_start:no_window_end]

    # 检查输入转换
    if 'view = view.permute(0, 2, 3, 1).contiguous()' in no_window_branch:
        # 检查是否无条件执行
        permute_line = no_window_branch.find('view = view.permute(0, 2, 3, 1).contiguous()')
        before_permute = no_window_branch[:permute_line]

        # 检查前面是否有 if self.using_real_ss2d
        if 'if self.using_real_ss2d:' in before_permute:
            print("❌ 输入转换仍然有条件判断（应该统一执行）")
        else:
            print("✅ 输入统一转换为 channels-last (B, H, W, C)")
    else:
        print("❌ 未找到输入格式转换")

    # 检查输出转换
    if 'out = out.permute(0, 3, 1, 2).contiguous()' in no_window_branch:
        # 检查是否无条件执行
        permute_line = no_window_branch.find('out = out.permute(0, 3, 1, 2).contiguous()')
        before_permute = no_window_branch[:permute_line]
        lines_before = before_permute.split('\n')

        # 检查这行前面最近的几行是否有 if
        recent_lines = '\n'.join(lines_before[-5:])
        if 'if self.using_real_ss2d:' in recent_lines:
            print("❌ 输出转换仍然有条件判断（应该统一执行）")
        else:
            print("✅ 输出统一转换为 channels-first (B, C, H, W)")
    else:
        print("❌ 未找到输出格式转换")

print("\n【检查项 2】局部滑窗版 (use_local_window=True)")
print("-"*80)

# 找到 else 分支（滑窗版）
window_branch_start = process_view_code.find('else:', no_window_end)
window_branch = process_view_code[window_branch_start:]

# 检查输入转换
if 'windows = windows.permute(0, 2, 3, 1).contiguous()' in window_branch:
    permute_line = window_branch.find('windows = windows.permute(0, 2, 3, 1).contiguous()')
    before_permute = window_branch[:permute_line]

    if 'if self.using_real_ss2d:' in before_permute:
        print("❌ 输入转换仍然有条件判断（应该统一执行）")
    else:
        print("✅ 窗口输入统一转换为 channels-last")
else:
    print("❌ 未找到窗口输入格式转换")

# 检查输出转换
if 'windows_out = windows_out.permute(0, 3, 1, 2).contiguous()' in window_branch:
    permute_line = window_branch.find('windows_out = windows_out.permute(0, 3, 1, 2).contiguous()')
    before_permute = window_branch[:permute_line]
    lines_before = before_permute.split('\n')
    recent_lines = '\n'.join(lines_before[-5:])

    if 'if self.using_real_ss2d:' in recent_lines:
        print("❌ 输出转换仍然有条件判断（应该统一执行）")
    else:
        print("✅ 窗口输出统一转换为 channels-first")
else:
    print("❌ 未找到窗口输出格式转换")

print("\n【检查项 3】是否移除了 using_real_ss2d 的条件判断")
print("-"*80)

# 在 _process_view 中查找 using_real_ss2d
if 'if self.using_real_ss2d:' in process_view_code:
    print("⚠️  _process_view 中仍有 using_real_ss2d 条件判断")
    print("   （应该统一处理，不再区分真实SS2D和fallback）")
else:
    print("✅ 已移除 using_real_ss2d 条件判断，统一处理格式")

print("\n【总结】")
print("-"*80)

# 汇总检查
checks = []

# 检查1：全局版输入
if ('view = view.permute(0, 2, 3, 1).contiguous()' in no_window_branch and
    'if self.using_real_ss2d:' not in no_window_branch[:no_window_branch.find('view = view.permute(0, 2, 3, 1).contiguous()')]):
    checks.append("✅ 全局版输入转换")
else:
    checks.append("❌ 全局版输入转换")

# 检查2：全局版输出
if 'out = out.permute(0, 3, 1, 2).contiguous()' in no_window_branch:
    checks.append("✅ 全局版输出转换")
else:
    checks.append("❌ 全局版输出转换")

# 检查3：滑窗版输入
if 'windows = windows.permute(0, 2, 3, 1).contiguous()' in window_branch:
    checks.append("✅ 滑窗版输入转换")
else:
    checks.append("❌ 滑窗版输入转换")

# 检查4：滑窗版输出
if 'windows_out = windows_out.permute(0, 3, 1, 2).contiguous()' in window_branch:
    checks.append("✅ 滑窗版输出转换")
else:
    checks.append("❌ 滑窗版输出转换")

for check in checks:
    print(check)

if all('✅' in check for check in checks):
    print("\n🎉 所有格式转换已统一处理！")
    print("   真实SS2D和fallback placeholder都使用 channels-last")
    print("   输入统一转换，输出统一转回")
else:
    print("\n⚠️  部分检查未通过，请检查代码")

print("="*80)
