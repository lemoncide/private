import ast
import operator
import math

def calculator(expression: str) -> float:
    """
    安全地计算一个数学表达式。

    何时用：需要进行确定性的数值计算（优先于让 LLM 口算）。
    输入：expression（表达式字符串；支持 + - * / ** % 及 sin/cos/sqrt 等函数）。
    输出：计算结果（float）。
    典型任务：单位换算、简单统计计算、阈值判断前的数值求解。
    """
    # Safe operators whitelist
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Safe functions whitelist
    functions = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "ceil": math.ceil,
        "floor": math.floor,
        "max": max,
        "min": min,
    }

    def eval_node(node):
        if isinstance(node, ast.Constant):  # Number or String
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        
        elif isinstance(node, ast.BinOp):  # <left> <op> <right>
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](eval_node(node.left), eval_node(node.right))
            raise ValueError(f"Unsupported operator: {op_type}")
            
        elif isinstance(node, ast.UnaryOp):  # <op> <operand> (e.g., -5)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](eval_node(node.operand))
            raise ValueError(f"Unsupported unary operator: {op_type}")
            
        elif isinstance(node, ast.Call):  # Function call e.g. sin(x)
            if isinstance(node.func, ast.Name) and node.func.id in functions:
                args = [eval_node(arg) for arg in node.args]
                return functions[node.func.id](*args)
            raise ValueError(f"Unsupported or unsafe function call: {node.func.id if isinstance(node.func, ast.Name) else node.func}")
            
        elif isinstance(node, ast.Expression):
            return eval_node(node.body)
            
        raise ValueError(f"Unsupported AST node: {type(node)}")

    try:
        # Parse expression to AST
        tree = ast.parse(expression, mode='eval')
        result = eval_node(tree.body)
        return float(result)
    except Exception as e:
        raise ValueError(f"Invalid expression '{expression}': {str(e)}")
