print("Its a match case calculator")
firstnum = int(input("Enter first number"))
operator = input("Enter operator( + , - , * , /)")
secnum= int(input("Enter second number"))

match operator:
    case "+":
        print(firstnum+secnum)
    case "-":
        print(firstnum-secnum)
    case "*":
        print(firstnum*secnum)
    case "/":
        print(firstnum/secnum)
    