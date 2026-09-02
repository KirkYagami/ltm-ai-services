

class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, my name is {self.name} and I am {self.age} years old."


data = {"p1":
        {
            "name": "Alice",
            "age": 30
        },
        "p2":
        {
            "name": "Bob",
            "age": 25
        }
        }

# p1 = Person(name = "Alice", age = 30)

for k, v in data.items():
    print(f"Creating Person object for {k} with data: {v}")
    person = Person(**v)
    print(person.greet())