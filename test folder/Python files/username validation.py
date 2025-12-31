print("username should be atleast 6 characters and should not include spaces")
username = input("Enter a username ")
totalchars=len(username)

def checker(x):
    if totalchars>6:
        if " " in username:
            return False
        else:
            return True
    else:
        return False

print(checker(username))