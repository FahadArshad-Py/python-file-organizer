num= int(input("Enter a number"))

def checker(x):
    if x%2==0:
        return True
    else:
        return False

if checker(num) is True:
    print("Even")
else:
    print("Odd")