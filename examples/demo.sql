-- Synthetic teaching data, unrelated to Spider evaluation results.
CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, department_id INTEGER, salary INTEGER,
                        FOREIGN KEY (department_id) REFERENCES departments(id));
INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Design');
INSERT INTO employees VALUES (1, 'Alice', 1, 12000), (2, 'Bob', 1, 10000), (3, 'Carol', 2, 11000);
