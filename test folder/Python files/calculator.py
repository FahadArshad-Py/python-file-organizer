print("It is a calculator")
firstnum = int(input("Enter first number"))
operator = input("Enter operator( + , - , * , /)")
secnum= int(input("Enter second number"))

def calculator(x,y):
    if operator == "+":
        return firstnum+secnum
    elif operator=="-":
        return firstnum-secnum
    elif operator=="*":
        return firstnum*secnum
    elif operator=="/":
        return firstnum/secnum
    else:
        return False

ans=calculator(firstnum,secnum)
print(f"Result: {ans}")    
