import textwrap


class SIRPrinter:
    def __init__(self):
        self.indent_ = 0
        self.T_ = textwrap.TextWrapper(initial_indent=' ' * self.indent_, width=120,
                                       subsequent_indent=' ' * self.indent_)

    def visit_builtin_type(self, builtin_type):
        if builtin_type.type_id == 0:
            raise ValueError('Builtin type not supported')
        elif builtin_type.type_id == 1:
            return "auto"
        elif builtin_type.type_id == 2:
            return "bool"
        elif builtin_type.type_id == 3:
            return "int"
        elif builtin_type.type_id == 4:
            return "float"
        raise ValueError('Builtin type not supported')

    def visit_unary_operator(self, expr):
        return expr.op + " " + self.visit_expr(expr.operand)

    def visit_binary_operator(self, expr):
        return "("+ self.visit_expr(expr.left) + " " + expr.op + " " + self.visit_expr(expr.right)+")"

    def visit_assignment_expr(self, expr):
        return self.visit_expr(expr.left) + " " + expr.op + " " + self.visit_expr(expr.right)

    def visit_ternary_operator(self, expr):
        return "(" + self.visit_expr(expr.cond) + " ? " + self.visit_expr(expr.left) + " : " + self.visit_expr(
            expr.right) + ")"

    def visit_var_access_expr(self, expr):
        return expr.name  # + self.visit_expr(expr.index)

    def visit_field_access_expr(self, expr):
        str_ = expr.name + "["
        if expr.WhichOneof("horizontal_offset") == "cartesian_offset":
            str_ += str(expr.cartesian_offset.i_offset) + ", "
            str_ += str(expr.cartesian_offset.j_offset)
        elif expr.WhichOneof("horizontal_offset") == "unstructured_offset":
            str_ += "<has_horizontal_offset>" if expr.unstructured_offset.has_offset else "<no_horizontal_offset>"
        elif expr.WhichOneof("horizontal_offset") == "zero_offset":
            str_ += "<no_horizontal_offset>"
        else:
            raise ValueError("Unknown offset")
        str_ += ", " + str(expr.vertical_offset)
        str_ += "]"
        return str_

    def visit_literal_access_expr(self, expr):
        return expr.value

    # call to external function, like math::sqrt
    def visit_fun_call_expr(self, expr):
        return expr.callee + "(" + ",".join(self.visit_expr(x) for x in expr.arguments) + ")"

    def visit_reduction_over_neighbor_expr(self, expr):
        return "reduce(" + expr.op + ", init=" + self.visit_expr(expr.init) + ", rhs=" \
            + self.visit_expr(expr.rhs) + ")"

    def visit_expr(self, expr):
        if expr.WhichOneof("expr") == "unary_operator":
            return self.visit_unary_operator(expr.unary_operator)
        elif expr.WhichOneof("expr") == "binary_operator":
            return self.visit_binary_operator(expr.binary_operator)
        elif expr.WhichOneof("expr") == "assignment_expr":
            return self.visit_assignment_expr(expr.assignment_expr)
        elif expr.WhichOneof("expr") == "ternary_operator":
            return self.visit_ternary_operator(expr.ternary_operator)
        elif expr.WhichOneof("expr") == "fun_call_expr":
            return self.visit_fun_call_expr(expr.fun_call_expr)
        elif expr.WhichOneof("expr") == "stencil_fun_call_expr":
            raise ValueError("non supported expression")
        elif expr.WhichOneof("expr") == "stencil_fun_arg_expr":
            raise ValueError("non supported expression")
        elif expr.WhichOneof("expr") == "var_access_expr":
            return self.visit_var_access_expr(expr.var_access_expr)
        elif expr.WhichOneof("expr") == "field_access_expr":
            return self.visit_field_access_expr(expr.field_access_expr)
        elif expr.WhichOneof("expr") == "literal_access_expr":
            return self.visit_literal_access_expr(expr.literal_access_expr)
        elif expr.WhichOneof("expr") == "reduction_over_neighbor_expr":
            return self.visit_reduction_over_neighbor_expr(expr.reduction_over_neighbor_expr)
        else:
            raise ValueError("Unknown expression")

    def visit_var_decl_stmt(self, var_decl):
        str_ = ""
        if var_decl.type.WhichOneof("type") == "name":
            str_ += var_decl.type.name
        elif var_decl.type.WhichOneof("type") == "builtin_type":
            str_ += self.visit_builtin_type(var_decl.type.builtin_type)
        else:
            raise ValueError("Unknown type ", var_decl.type.WhichOneof("type"))
        str_ += " " + var_decl.name

        if var_decl.dimension != 0:
            str_ += "[" + str(var_decl.dimension) + "]"

        str_ += var_decl.op

        for expr in var_decl.init_list:
            str_ += self.visit_expr(expr)

        print(self.T_.fill(str_))

    def visit_expr_stmt(self, stmt):
        print(self.T_.fill(self.visit_expr(stmt.expr)))

    def visit_if_stmt(self, stmt):
        cond = stmt.cond_part
        if cond.WhichOneof("stmt") != "expr_stmt":
            raise ValueError("Not expected stmt")

        print(self.T_.fill("if(" + self.visit_expr(cond.expr_stmt.expr) + ")"))
        self.visit_body_stmt(stmt.then_part)
        self.visit_body_stmt(stmt.else_part)

    def visit_block_stmt(self, stmt):
        print(self.T_.fill("{"))
        self.indent_ += 2
        self.T_.initial_indent = ' ' * self.indent_

        for each in stmt.statements:
            self.visit_body_stmt(each)
        self.indent_ -= 2
        self.T_.initial_indent = ' ' * self.indent_

        print(self.T_.fill("}"))

    def visit_body_stmt(self, stmt):
        if stmt.WhichOneof("stmt") == "var_decl_stmt":
            self.visit_var_decl_stmt(stmt.var_decl_stmt)
        elif stmt.WhichOneof("stmt") == "expr_stmt":
            self.visit_expr_stmt(stmt.expr_stmt)
        elif stmt.WhichOneof("stmt") == "if_stmt":
            self.visit_if_stmt(stmt.if_stmt)
        elif stmt.WhichOneof("stmt") == "block_stmt":
            self.visit_block_stmt(stmt.block_stmt)
        else:
            raise ValueError("Stmt not supported :" + stmt.WhichOneof("stmt"))

    def visit_vertical_region(self, vertical_region):
        str_ = "vertical_region("
        interval = vertical_region.interval
        if (interval.WhichOneof("LowerLevel") == 'special_lower_level'):
            if interval.special_lower_level == 0:
                str_ += "kstart"
            else:
                str_ += "kend"
        elif (interval.WhichOneof("LowerLevel") == 'lower_level'):
            str_ += str(interval.lower_level)
        if interval.lower_offset != 0:
            str_ += "+"+str(interval.lower_offset)
        str_ += ","
        if (interval.WhichOneof("UpperLevel") == 'special_upper_level'):
            if interval.special_upper_level == 0:
                str_ += "kstart"
            else:
                str_ += "kend"
        elif (interval.WhichOneof("UpperLevel") == 'upper_level'):
            str_ += str(interval.upper_level)
        if interval.upper_offset != 0:
            if(interval.upper_offset >0 ):
              str_ += "+"+str(interval.upper_offset)
            else:
              str_ += "-" + str(-interval.upper_offset)
        str_ += ")"
        print(self.T_.fill(str_))

        self.indent_ += 2
        self.T_.initial_indent = ' ' * self.indent_

        for stmt in vertical_region.ast.root.block_stmt.statements:
            self.visit_body_stmt(stmt)

        self.indent_ -= 2
        self.T_.initial_indent = ' ' * self.indent_

    def visit_stencil(self, stencil):
        print(self.T_.fill("stencil " + stencil.name))
        self.indent_ += 2
        self.T_.initial_indent = ' ' * self.indent_

        self.visit_fields(stencil.fields)

        block = stencil.ast.root.block_stmt
        for stmt in block.statements:
            if stmt.WhichOneof("stmt") == "vertical_region_decl_stmt":
                self.visit_vertical_region(stmt.vertical_region_decl_stmt.vertical_region)

        self.indent_ -= 2
        self.T_.initial_indent = ' ' * self.indent_

    def visit_global_variables(self, global_variables):
        if not global_variables.IsInitialized():
            return

        print(self.T_.fill("globals"))

        self.indent_ += 2
        self.T_.initial_indent = ' ' * self.indent_
        for name, value in global_variables.map.iteritems():
            str_ = ""
        if value.WhichOneof("Value") == "integer_value":
            str_ += "integer " + name
            if (value.is_constexpr):
                str_ += "= " + value.integer_value

        if value.WhichOneof("Value") == "double_value":
            str_ += "double " + name
            if (value.is_constexpr):
                str_ += "= " + value.double_value

        if value.WhichOneof("Value") == "boolean_value":
            str_ += "bool " + name
            if (value.is_constexpr):
                str_ += "= " + value.boolean_value

        if value.WhichOneof("Value") == "string_value":
            str_ += "string " + name
            if (value.is_constexpr):
                str_ += "= " + value.string_value

        print(self.T_.fill(str_))
        self.indent_ -= 2
        self.T_.initial_indent = ' ' * self.indent_

    def visit_fields(self, fields):
        str_ = "field "
        for field in fields:
            str_ += field.name
            dims = ["i", "j", "k"]
            dims_ = []
            for i in range(0, 3):
                if field.field_dimensions[i] != -1:
                    dims_.append(dims[i])
            str_ += str(dims_)
            str_ += ","
        print(self.T_.fill(str_))

