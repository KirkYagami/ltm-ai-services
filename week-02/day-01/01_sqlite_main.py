import sqlite3

# Create an in-memory SQLite database
conn = sqlite3.connect(":memory:")
cursor = conn.cursor()

# Create students table
cursor.execute("""
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    student_name TEXT,
    department TEXT
)
""")




# Create grades table
cursor.execute("""
CREATE TABLE grades (
    student_id INTEGER,
    subject TEXT,
    marks INTEGER,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
)
""")

# Insert student data
students = [
    (1, "Amit", "CSE"),
    (2, "Neha", "CSE"),
    (3, "Raj", "ECE"),
    (4, "Sara", "EEE"),
    (5, "Vikram", "CSE")
]

cursor.executemany("""
INSERT INTO students (student_id, student_name, department)
VALUES (?, ?, ?)
""", students)



# Insert grade data
grades = [
    (1, "Math", 85),
    (1, "Physics", 78),
    (1, "Chemistry", 90),

    (2, "Math", 72),
    (2, "Physics", 69),
    (2, "Chemistry", 80),

    (3, "Math", 88),
    (3, "Physics", 92),
    (3, "Chemistry", 81),

    (4, "Math", 60),
    (4, "Physics", 58),
    (4, "Chemistry", 65),

    (5, "Math", 88),
    (5, "Physics", 87),
    (5, "Chemistry", 88)
]


cursor.executemany("""
INSERT INTO grades (student_id, subject, marks)
VALUES (?, ?, ?)
""", grades)

conn.commit()




sql_query = """
select s.student_name, 

ROUND(AVG(g.marks),2) as 
avg_marks 

from students s
INNER JOIN
grades g
ON s.student_id = g.student_id

GROUP BY s.student_id, s.student_name
ORDER BY avg_marks DESC
"""

res = cursor.execute(sql_query).fetchall()

for row in res:
    print(row)



cursor.close()

conn.close()
