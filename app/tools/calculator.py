"""安全计算器 — 基于 AST 白名单的算术表达式求值

为什么不用 eval()：
    eval("__import__('os').system('rm -rf /')")  ← 一句话删全盘
    AST 白名单只允许安全节点，恶意代码直接拒绝

安全原理：
    1. 把用户输入的字符串解析成抽象语法树（AST）
    2. 遍历 AST 节点，只允许数字、四则运算、幂运算
    3. 遇到任何不在白名单的节点类型 → 拒绝
"""

import ast
import math
import operator


# 白名单：只允许这些 Python 操作符和函数
_ALLOWED_OPS = {
    ast.Add: operator.add,       # +
    ast.Sub: operator.sub,       # -
    ast.Mult: operator.mul,      # *
    ast.Div: operator.truediv,   # /
    ast.FloorDiv: operator.floordiv,  # //
    ast.Mod: operator.mod,       # %
    ast.Pow: operator.pow,       # **
    ast.USub: operator.neg,      # 负号
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
}


def _eval_node(node: ast.AST) -> float:
    """递归求值 AST 节点，不在白名单则拒绝"""
    # 数字字面量：3.14, 42
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)

    # 二元运算：a + b, a * b 等
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPS[type(node.op)](left, right)  # pyright: ignore[reportArgumentType,reportCallIssue,reportUnknownVariableType]

    # 一元运算：-x
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        operand = _eval_node(node.operand)
        return _ALLOWED_OPS[type(node.op)](operand)  # pyright: ignore[reportArgumentType,reportCallIssue]

    # 允许的安全函数调用：abs(-5), sqrt(4)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in _ALLOWED_FUNCTIONS and len(node.args) == 1:
            arg = _eval_node(node.args[0])
            return _ALLOWED_FUNCTIONS[node.func.id](arg)

    raise ValueError(f"表达式包含不安全的操作: {ast.dump(node)}")


def calculate(expression: str) -> float:
    """
    安全计算表达式

    示例：
        calculate("3 + 4 * 2")    → 11.0
        calculate("sqrt(4)")      → 2.0
        calculate("abs(-5)")      → 5.0

    参数：
        expression : 算术表达式字符串

    返回：
        计算结果（float）

    异常：
        ValueError / SyntaxError : 表达式非法或不安全
    """
    # 1. 字符串 → AST
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise SyntaxError(f"表达式语法错误: {expression}") from None

    # 2. AST → 值（在白名单约束下安全求值）
    return _eval_node(tree.body)



