subjects= {"maths":90, "English":75, "Science":60}
total=0
for x in subjects:
    total += subjects[x]
percent=total/len(subjects)
print(total)
print(percent)
if percent>=50:
    print("PASS")
else:
    print("FAIL")
