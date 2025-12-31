
choice = input("Enter 'W' for withdraw OR 'D' for deposit")
amount = int(input("Enter amount"))

def bank():
    balance = 1000
    if choice=="W" or choice =="w":
        if amount > balance:
            print("insuficient balance!")
        else:
            balance = balance-amount
            print(f"{amount} is widthrawn")
            print(f"remaining balance is: {balance}")
    elif choice=="D" or choice=="d":
        balance = balance + amount
        print(f"{amount} is deposited")
        print(f"new balance is: {balance}")
    else:
        print("invalid choice!")

bank()