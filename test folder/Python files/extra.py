students = [
    {"name": "Ali", "courses": ["Python", "Java"]},
    {"name": "Sara", "courses": ["Python", "C++"]},
    {"name": "Ahmed", "courses": ["Java", "JavaScript"]}
]


for x in students:
    if x['name']=='Ali':
        for c in x['courses']:
            print(c)


