contacts={"Ali": "03001234567", "Fahad": "03111223344"}
def add(fname,fnum):
    contacts.update({fname:fnum})
    print("contact added successfully!")
    
def search(fname):
    if fname in contacts:
        x=contacts[fname]
        print("contact found")
        print(x)
    else:
        print("NOt found!")
def listall():
    for x in contacts:
        print(x,contacts[x])
while True:
    print("Press '1' to Add new contact")
    print("Press '2' to search contact")
    print("Press '3' to list all contacts")
    print("Press '4' to exit")
    choice=input("Enter your choice")

    if choice=="1":
        name=input("Enter name of contact")
        num=input("Enter number for contact")
        add(name,num)
    elif choice=="2":
        name=input("Enter name to search")
        search(name)
    elif choice=="3":
        listall()
    elif choice=="4":
        break
    else:
        print("Invalid choice!")