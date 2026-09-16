from engine import Value

a = Value(2.0)
b = Value(-3)
c = Value(10)

print(f"a = {a}")
print(f"b = {b}")

print(f"a + b = {a + b}")
print(f"a * b = {a * b}")

print()

d = a*b + c
print(f"d = {d}")
print(f"d._prev = {d._prev}")
print(f"d._op = {d._op}")
