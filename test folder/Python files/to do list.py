print("Welcome to your TO DO List")
to_do=[]

while True:
    print("Press '1' to Add a task")
    print("Press '2' to Remove a task")
    print("Press '3' to View all tasks")
    print("Press '4' to Exit")

    choice=input("Enter your choice")

    if choice=="1":
        print("Enter task")
        task=input()
        to_do.append(task)
        print("Task Added!")
    elif choice=="2":
        print("All tasks")
        i=0
        while i < len(to_do):
            print(i+1,to_do[i])
            i+=1
        num=int(input("Enter number to delete task"))
        index=num-1
        to_do.pop(index)
        print("Task removed")
    elif choice=="3":
        print("All tasks")
        i=0
        while i < len(to_do):
            print(i+1,to_do[i])
            i+=1
    elif choice=="4":
        break
    else:
        print("invalid choice")

