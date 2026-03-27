# ########################################################################################


# class _AST_to_MathExpr(ConversionRule):
#     def generic_map(self, model: Any, operands):
#         raise TypeError

#     def map_Module(self, model: ast.Module, operands):
#         if len(model.body) == 1:
#             return self(model.body[0])
#         raise TypeError

#     def map_Expr(self, model: ast.Expr, operands):
#         return self(model.value)

#     def map_Constant(self, model: ast.Constant, operands):
#         return MathExpr.cast(model.value)

#     def map_Name(self, model: ast.Name, operands):
#         return MathVar(name=model.id)

#     def map_BinOp(self, model: ast.BinOp, operands):
#         if isinstance(model.op, ast.Add):
#             return MathAdd(expr1=self(model.left), expr2=self(model.right))
#         if isinstance(model.op, ast.Sub):
#             return MathSub(expr1=self(model.left), expr2=self(model.right))
#         if isinstance(model.op, ast.Mult):
#             return MathMul(expr1=self(model.left), expr2=self(model.right))
#         if isinstance(model.op, ast.Div):
#             return MathDiv(expr1=self(model.left), expr2=self(model.right))
#         if isinstance(model.op, ast.Pow):
#             return MathPow(expr1=self(model.left), expr2=self(model.right))
#         raise TypeError

#     def map_UnaryOp(self, model: ast.UnaryOp, operands):
#         if isinstance(model.op, ast.USub):
#             return -self(model.operand)
#         if isinstance(model.op, ast.UAdd):
#             return self(model.operand)
#         raise TypeError

#     def map_Call(self, model: ast.Call, operands):
#         return MathFunc(
#             func=model.func.id,
#             expr=[self(arg) for arg in model.args],
#         )


# def MathStr(*, string):
#     return _AST_to_MathExpr()(ast.parse(string))

########################################################################################


# class _MathExprIsConstant(RewriteRule):
#     def map_MathExpr(self, model):
#         if getattr(self, "isconstant", None) is None:
#             self.isconstant = True

#     def map_MathVar(self, model):
#         self.isconstant = False

#     def map_Access(self, model):
#         self.isconstant = False


# def _isconstant(model):
#     constant_analysis = _MathExprIsConstant()

#     Post(constant_analysis)(model)

#     if constant_analysis.isconstant:
#         return model

#     raise ValueError("MathExpr is not a constant")


# ConstantMathExpr = Annotated[
#     CastMathExpr,
#     AfterValidator(_isconstant),
# ]
# """
# Annotated type for constant MathExpr
# """
