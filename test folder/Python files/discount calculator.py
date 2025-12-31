price =int(input("Enter product price"))

if price>=10000:
    dcprice=price-(price*0.2)
    print(f"Final price after discount = {dcprice}")
elif price>=5000 and price<10000:
    dcprice=price-(price*0.1)
    print(f"Final price after discount = {dcprice}")
else:
    print("no discount")