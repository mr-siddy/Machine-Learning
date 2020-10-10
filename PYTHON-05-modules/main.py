import calculator

# from calculator import add (
#    add as addition
#   sub as substraction
# )



print("hello world")

# if we are using import calculator
# we can not call add directly because add function is packed in module calculator

print(calculator.add(1,2))

# if we use from calculator import add -->> we can directly add

#print(add(1, 2))


def func():
    print("hello")

func()



print(calculator.CONSTANT)


