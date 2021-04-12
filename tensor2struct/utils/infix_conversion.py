# Adapted from https://gist.github.com/awadalaa/7ef7dc7e41edb501d44d1ba41cbf0dc6
class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[self.size() - 1]

    def size(self):
        return len(self.items)


class InfixConverter:
    def __init__(self):
        self.stack = Stack()
        self.precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}

    def hasLessOrEqualPriority(self, a, b):
        if a not in self.precedence:
            return False
        if b not in self.precedence:
            return False
        return self.precedence[a] <= self.precedence[b]

    def isOperator(self, x):
        ops = ["+", "-", "/", "*"]
        return x in ops

    def isOperand(self, ch):
        return ch.isalpha() or ch.isdigit()

    def isOpenParenthesis(self, ch):
        return ch == "("

    def isCloseParenthesis(self, ch):
        return ch == ")"

    def toPostfix(self, expr):
        expr = expr.replace(" ", "")
        self.stack = Stack()
        output = ""

        for c in expr:
            if self.isOperand(c):
                output += c
            else:
                if self.isOpenParenthesis(c):
                    self.stack.push(c)
                    output += c
                elif self.isCloseParenthesis(c):
                    operator = self.stack.pop()
                    while not self.isOpenParenthesis(operator):
                        output += operator
                        operator = self.stack.pop()
                    output += c
                else:
                    while (not self.stack.isEmpty()) and self.hasLessOrEqualPriority(
                        c, self.stack.peek()
                    ):
                        output += self.stack.pop()
                    self.stack.push(c)

        while not self.stack.isEmpty():
            output += self.stack.pop()
        return output

    """
     1. Reverse expression string
     2. Replace open paren with close paren and vice versa
     3. Get Postfix and reverse it
    """

    def _reverse(self, expr):
        reverse_expr = ""
        for c in expr[::-1]:
            if c == "(":
                reverse_expr += ")"
            elif c == ")":
                reverse_expr += "("
            else:
                reverse_expr += c
        return reverse_expr

    def _reverse_p(self, expr):
        reverse_p = ""
        for c in expr:
            if c == "(":
                reverse_p += ")"
            elif c == ")":
                reverse_p += "("
            else:
                reverse_p += c
        return reverse_p

    def toPrefix(self, expr):
        reverse_expr = self._reverse(expr)
        reverse_postfix = self.toPostfix(reverse_expr)
        real_postfix = self._reverse_p(reverse_postfix[::-1])
        return real_postfix

    def convert(self, expr):
        try:
            result = eval(expr)
        except:
            result = expr
        print(
            """
            Original expr is: {}
            Postfix is: {}
            Prefix is: {}
            result is: {}
        """.format(
                expr, self.toPostfix(expr), self.toPrefix(expr), result
            )
        )
