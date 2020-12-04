def interp(code):
    data = [0 for i in range(10000)]
    pc = 0
    ptr = 0
    skip_loop = False
    bracket_count = 0
    stack = []
    while pc < len(code):
        c = code[pc]
        if skip_loop:
            if c == '[':
                bracket_count += 1
            elif c == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    skip_loop = False
            pc += 1
            continue
        if c == '>':
            ptr += 1
            pc += 1
        elif c == '<':
            ptr -= 1
            pc += 1
        elif c == '+':
            data[ptr] += 1
            pc += 1
        elif c == '-':
            data[ptr] -= 1
            pc += 1
        elif c == '.':
            print(chr(data[ptr]), end="")
            pc += 1
        elif c == ',':
            pc += 1
        elif c == '[':
            if data[ptr] == 0:
                bracket_count = 1
                skip_loop = True
                pc +=1
            else:
                pc += 1
                stack.append(pc)
        elif c == ']':
            if data[ptr] == 0:
                pc += 1
                stack.pop()
            else:
                pc = stack[len(stack) - 1]


interp(
    '++++++++++[>+++++++>++++++++++>+++>+<<<<-]>++.>+.+++++++..+++.>++.<<+++++++++++++++.>.+++.------.--------.>+.>.')
