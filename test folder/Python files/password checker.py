password=input("Enter a passsword")

def pass_check(fpass):
    len_ok=False
    one_digit=False
    upper_char=False
    if len(fpass)>=8:
        len_ok=True
    for x in fpass:
        if x.isdigit():
            one_digit=True
        if x.upper():
            upper_char=True
    if len_ok and one_digit and upper_char:
        print("strong password")
    else:
        print("weak password")

pass_check(password)   