# SQL Interview Questions and Answers - Complete Guide

## Table of Contents
- [JOIN Operations](#join-operations)
- [Window Functions](#window-functions)
- [Aggregation Functions](#aggregation-functions)
- [Subqueries](#subqueries)
- [Data Manipulation](#data-manipulation)
- [Performance & Optimization](#performance--optimization)
- [Advanced SQL Concepts](#advanced-sql-concepts)
- [Common Interview Scenarios](#common-interview-scenarios)
- [Stored Procedures (SPs)](#stored-procedures-sps)
- [User Defined Functions (UDFs)](#user-defined-functions-udfs)
- [Quick Tips for SQL Interviews](#quick-tips-for-sql-interviews)

## JOIN Operations

### Q1: What are the different types of JOINs in SQL?
**Answer**: JOINs combine rows from two or more tables based on a condition. The main types are:
- **INNER JOIN**: Only shows rows where there's a match in both tables, like finding common friends between two groups.
- **LEFT JOIN**: Shows all rows from the left table, even if there's no match in the right table (unmatched right table columns get NULL).
- **RIGHT JOIN**: Shows all rows from the right table, even if there's no match in the left table (unmatched left table columns get NULL).
- **FULL OUTER JOIN**: Shows all rows from both tables, with NULLs where there's no match, like combining two lists completely.
- **CROSS JOIN**: Creates all possible combinations of rows from both tables, like pairing every shirt with every pant.

### Q2: Write a query to find employees and their department names using INNER JOIN
```sql
SELECT e.employee_name, d.department_name
FROM employees e
INNER JOIN departments d ON e.department_id = d.department_id;
```
**Explanation**: This query matches employees with their departments, showing only employees who are assigned to a valid department (no unmatched rows).

### Q3: What's the difference between INNER JOIN and LEFT JOIN?
**Answer**:
- **INNER JOIN**: Only includes rows where both tables have matching values, like finding students enrolled in specific classes.
- **LEFT JOIN**: Includes all rows from the left table, even if there's no match in the right table, filling in NULLs for missing matches, like listing all students whether they’re enrolled or not.

### Q4: Find employees without any department assigned
```sql
SELECT e.employee_name
FROM employees e
LEFT JOIN departments d ON e.department_id = d.department_id
WHERE d.department_id IS NULL;
```
**Explanation**: This uses a LEFT JOIN to include all employees and checks for NULLs in the department table to find employees not assigned to any department.

### Q5: Write a self-join query to find employees and their managers
```sql
SELECT e.employee_name, m.employee_name as manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id;
```
**Explanation**: A self-join treats the same table (employees) as two tables to link each employee to their manager based on the manager_id.

## Window Functions

### Q6: What are window functions in SQL?
**Answer**: Window functions calculate values across a group of rows related to the current row without combining them into one row (unlike GROUP BY). They’re like looking at a subset of data for each row while keeping all rows visible.

### Q7: Write a query to rank employees by salary within each department
```sql
SELECT employee_name, department_id, salary,
       RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as salary_rank
FROM employees;
```
**Explanation**: This ranks employees within each department based on their salary, with the highest salary getting rank 1 in each department.

### Q8: What's the difference between RANK(), DENSE_RANK(), and ROW_NUMBER()?
**Answer**:
- **ROW_NUMBER()**: Gives each row a unique number, like 1, 2, 3, 4, 5, no matter if values are tied.
- **RANK()**: Gives tied values the same rank but skips the next rank(s), like 1, 2, 2, 4, 5.
- **DENSE_RANK()**: Gives tied values the same rank without skipping, like 1, 2, 2, 3, 4.

### Q9: Find the 2nd highest salary in each department
```sql
SELECT department_id, employee_name, salary
FROM (
    SELECT employee_name, department_id, salary,
           DENSE_RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) as rn
    FROM employees
) ranked
WHERE rn = 2;
```
**Explanation**: This assigns a rank to salaries within each department and selects rows where the rank is 2, giving the second-highest salary.

### Q10: Calculate running total of sales by month
```sql
SELECT month, sales,
       SUM(sales) OVER (ORDER BY month) as running_total
FROM monthly_sales;
```
**Explanation**: This calculates a cumulative sum of sales for each month, adding up all previous months’ sales.

### Q11: Find employees with salary above department average
```sql
SELECT employee_name, salary, department_id
FROM (
    SELECT employee_name, salary, department_id,
           AVG(salary) OVER (PARTITION BY department_id) as dept_avg
    FROM employees
) e
WHERE salary > dept_avg;
```
**Explanation**: This compares each employee’s salary to the average salary of their department, showing only those above the average.

## Aggregation Functions

### Q12: What are the main aggregate functions?
**Answer**:
- **COUNT()**: Counts how many rows exist, like tallying attendees at an event.
- **SUM()**: Adds up numeric values, like totaling sales.
- **AVG()**: Calculates the average of values, like finding average test scores.
- **MAX()**: Finds the highest value, like the most expensive item.
- **MIN()**: Finds the lowest value, like the cheapest item.

### Q13: Find department with highest average salary
```sql
SELECT department_id, AVG(salary) as avg_salary
FROM employees
GROUP BY department_id
ORDER BY avg_salary DESC
LIMIT 1;
```
**Explanation**: This groups employees by department, calculates the average salary per department, and picks the one with the highest average.

### Q14: What's the difference between COUNT(*) and COUNT(column)?
**Answer**:
- **COUNT(*)**: Counts every row, even if it has NULLs, like counting all tickets sold.
- **COUNT(column)**: Counts only rows where the specified column isn’t NULL, like counting only tickets with a buyer’s name.

### Q15: Find departments with more than 5 employees
```sql
SELECT department_id, COUNT(*) as employee_count
FROM employees
GROUP BY department_id
HAVING COUNT(*) > 5;
```
**Explanation**: This groups employees by department and filters for departments with more than 5 employees.

### Q16: Calculate total sales by year and quarter
```sql
SELECT YEAR(order_date) as year, 
       QUARTER(order_date) as quarter,
       SUM(amount) as total_sales
FROM orders
GROUP BY YEAR(order_date), QUARTER(order_date)
ORDER BY year, quarter;
```
**Explanation**: This groups orders by year and quarter, summing the sales amounts for each group.

## Subqueries

### Q17: What are subqueries? Types of subqueries?
**Answer**: Subqueries are queries nested inside another query, like a question within a question. Types:
- **Scalar subquery**: Returns one value, like an average.
- **Row subquery**: Returns one row with multiple columns.
- **Column subquery**: Returns one column with multiple rows, like a list of IDs.
- **Table subquery**: Returns multiple rows and columns, like a mini-table.

### Q18: Find employees earning more than average salary
```sql
SELECT employee_name, salary
FROM employees
WHERE salary > (SELECT AVG(salary) FROM employees);
```
**Explanation**: This compares each employee’s salary to the overall average salary calculated in the subquery.

### Q19: Find departments with no employees
```sql
SELECT department_name
FROM departments
WHERE department_id NOT IN (
    SELECT DISTINCT department_id 
    FROM employees 
    WHERE department_id IS NOT NULL
);
```
**Explanation**: This finds departments that don’t appear in the employees table, meaning they have no employees.

### Q20: Find employees in departments with highest total salary
```sql
SELECT employee_name, department_id
FROM employees
WHERE department_id = (
    SELECT department_id
    FROM employees
    GROUP BY department_id
    ORDER BY SUM(salary) DESC
    LIMIT 1
);
```
**Explanation**: This identifies the department with the highest total salary in a subquery, then lists all employees in that department.

## Data Manipulation

### Q21: Write INSERT statement with multiple rows
```sql
INSERT INTO employees (employee_name, department_id, salary)
VALUES 
    ('John Doe', 1, 50000),
    ('Jane Smith', 2, 60000),
    ('Bob Johnson', 1, 55000);
```
**Explanation**: This adds multiple employee records at once, like filling out several forms in one go.

### Q22: Update salary of all employees in a specific department
```sql-UPDATE employees 
SET salary = salary * 1.10 
WHERE department_id = 1;
```
**Explanation**: This increases salaries by 10% for all employees in department 1.

### Q23: Delete employees with salary less than 30000
```sql
DELETE FROM employees 
WHERE salary < 30000;
```
**Explanation**: This removes all employee records with salaries below 30,000.

### Q24: Create a backup table with data
```sql
CREATE TABLE employees_backup AS
SELECT * FROM employees;
```
**Explanation**: This creates a new table with the same data as the employees table, like making a photocopy.

## Performance & Optimization

### Q25: What is an index? Types of indexes?
**Answer**: An index is like a book’s index, speeding up searches by organizing data. Types:
- **Clustered**: Physically sorts table data, like arranging books alphabetically.
- **Non-clustered**: A separate lookup table pointing to data, like a library catalog.
- **Unique**: Ensures no duplicate values, like unique IDs.
- **Composite**: Indexes multiple columns together, like a phonebook with name and city.

### Q26: When should you use indexes?
**Answer**: Use indexes for:
- Columns you search often, like looking up names.
- Columns used in JOINs, like linking tables.
- Columns in ORDER BY, like sorting results.
Don’t overuse them, as they slow down INSERT, UPDATE, and DELETE operations.

### Q27: What is a query execution plan?
**Answer**: A query execution plan is like a GPS route for a query, showing how the database will process it, including which indexes to use, join methods, and estimated costs.

### Q28: How to optimize a slow query?
**Answer**:
- Add indexes on frequently searched or joined columns.
- Avoid `SELECT *`; pick only needed columns.
- Use precise WHERE conditions to limit rows.
- Optimize JOINs with proper conditions.
- Use LIMIT to reduce output rows.
- Check the execution plan to spot bottlenecks.

## Advanced SQL Concepts

### Q29: What are CTEs (Common Table Expressions)?
**Answer**: CTEs are like temporary mini-tables created for one query, making complex queries easier to read and reuse.

```sql
WITH dept_avg AS (
    SELECT department_id, AVG(salary) as avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT e.employee_name, e.salary, d.avg_salary
FROM employees e
JOIN dept_avg d ON e.department_id = d.department_id;
```
**Explanation**: This creates a temporary table (dept_avg) with average salaries per department, then joins it with employees to compare salaries.

### Q30: Write a recursive CTE to find employee hierarchy
```sql
WITH RECURSIVE employee_hierarchy AS (
    -- Base case: top-level managers
    SELECT employee_id, employee_name, manager_id, 1 as level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees with managers
    SELECT e.employee_id, e.employee_name, e.manager_id, eh.level + 1
    FROM employees e
    JOIN employee_hierarchy eh ON e.manager_id = eh.employee_id
)
SELECT * FROM employee_hierarchy;
```
**Explanation**: This builds an employee hierarchy, starting with top managers and recursively adding their direct reports, tracking the level in the hierarchy.

### Q31: What are CASE statements?
**Answer**: CASE statements are like if-then-else logic in SQL, letting you categorize data based on conditions.

```sql
SELECT employee_name, salary,
    CASE 
        WHEN salary > 70000 THEN 'High'
        WHEN salary > 50000 THEN 'Medium'
        ELSE 'Low'
    END as salary_category
FROM employees;
```
**Explanation**: This labels salaries as High, Medium, or Low based on their value.

### Q32: Find duplicate records
```sql
SELECT employee_name, COUNT(*)
FROM employees
GROUP BY employee_name
HAVING COUNT(*) > 1;
```
**Explanation**: This groups employees by name and shows names that appear more than once, indicating duplicates.

### Q33: Remove duplicate records keeping only one
```sql
DELETE e1 FROM employees e1
INNER JOIN employees e2 
WHERE e1.employee_id > e2.employee_id 
AND e1.employee_name = e2.employee_name;
```
**Explanation**: This deletes duplicate employee records, keeping the one with the lower employee_id.

### Q34: What is the difference between UNION and UNION ALL?
**Answer**:
- **UNION**: Combines results from two queries, removing duplicates, which is slower.
- **UNION ALL**: Combines results without removing duplicates, which is faster.

### Q35: Pivot table example - convert rows to columns
```sql
SELECT department_id,
    SUM(CASE WHEN YEAR(hire_date) = 2022 THEN 1 ELSE 0 END) as "2022",
    SUM(CASE WHEN YEAR(hire_date) = 2023 THEN 1 ELSE 0 END) as "2023",
    SUM(CASE WHEN YEAR(hire_date) = 2024 THEN 1 ELSE 0 END) as "2024"
FROM employees
GROUP BY department_id;
```
**Explanation**: This turns hire years into columns, counting how many employees were hired in each year per department.

### Q36: LAG and LEAD functions
```sql
SELECT employee_name, salary,
    LAG(salary) OVER (ORDER BY salary) as prev_salary,
    LEAD(salary) OVER (ORDER BY salary) as next_salary
FROM employees;
```
**Explanation**: This shows each employee’s salary alongside the previous and next salaries when ordered by salary.

### Q37: NTILE function for quartiles
```sql
SELECT employee_name, salary,
    NTILE(4) OVER (ORDER BY salary) as quartile
FROM employees;
```
**Explanation**: This divides employees into four equal groups (quartiles) based on their salary.

### Q38: What are constraints in SQL?
**Answer**:
- **PRIMARY KEY**: Uniquely identifies each row, like a student ID.
- **FOREIGN KEY**: Links to a primary key in another table, like linking orders to customers.
- **UNIQUE**: Prevents duplicate values, like unique usernames.
- **NOT NULL**: Requires a value, like a mandatory name field.
- **CHECK**: Enforces a condition, like age > 18.

### Q39: What is normalization?
**Answer**: Normalization organizes data to avoid duplication, like splitting a big spreadsheet into smaller, linked tables:
- **1NF**: No repeating groups, all values are single (atomic).
- **2NF**: 1NF + no partial dependencies (all columns depend on the full primary key).
- **3NF**: 2NF + no transitive dependencies (no column depends on another non-key column).

### Q40: What are triggers?
**Answer**: Triggers are automatic actions that run when data changes (INSERT, UPDATE, DELETE), like auto-updating a timestamp.

```sql
CREATE TRIGGER update_modified_date
BEFORE UPDATE ON employees
FOR EACH ROW
SET NEW.modified_date = NOW();
```
**Explanation**: This sets the modified_date to the current time whenever an employee record is updated.

## Common Interview Scenarios

### Q41: Find Nth highest salary
```sql
-- Method 1: Using LIMIT and OFFSET
SELECT DISTINCT salary
FROM employees
ORDER BY salary DESC
LIMIT 1 OFFSET (N-1);

-- Method 2: Using subquery
SELECT salary
FROM employees e1
WHERE (N-1) = (
    SELECT COUNT(DISTINCT salary)
    FROM employees e2
    WHERE e2.salary > e1.salary
);
```
**Explanation**: These methods find the Nth highest salary, either by skipping (N-1) salaries or by counting how many salaries are higher.

### Q42: Find employees hired in last 30 days
```sql
SELECT employee_name, hire_date
FROM employees
WHERE hire_date >= CURDATE() - INTERVAL 30 DAY;
```
**Explanation**: This lists employees hired within the last 30 days from today.

### Q43: Calculate age from birthdate
```sql
SELECT employee_name, birthdate,
    TIMESTAMPDIFF(YEAR, birthdate, CURDATE()) as age
FROM employees;
```
**Explanation**: This calculates each employee’s age based on their birthdate and today’s date.

### Q44: Find gaps in sequence
```sql
SELECT (t1.id + 1) as gap_start,
    (SELECT MIN(t2.id) - 1 FROM sequence_table t2 WHERE t2.id > t1.id) as gap_end
FROM sequence_table t1
WHERE NOT EXISTS (SELECT 1 FROM sequence_table t2 WHERE t2.id = t1.id + 1)
AND t1.id < (SELECT MAX(id) FROM sequence_table);
```
**Explanation**: This finds missing numbers in a sequence, like gaps in a list of IDs.

### Q45: Convert comma-separated values to rows
```sql
-- Using recursive CTE (MySQL 8.0+)
WITH RECURSIVE split_strings AS (
    SELECT 
        id,
        SUBSTRING_INDEX(tags, ',', 1) as tag,
        CASE WHEN LOCATE(',', tags) > 0 
             Ascendancy
             THEN SUBSTRING(tags, LOCATE(',', tags) + 1)
             ELSE NULL END as remaining
    FROM products
    WHERE tags IS NOT NULL
    
    UNION ALL
    
    SELECT 
        id,
        SUBSTRING_INDEX(remaining, ',', 1),
        CASE WHEN LOCATE(',', remaining) > 0 
             THEN SUBSTRING(remaining, LOCATE(',', remaining) + 1)
             ELSE NULL END
    FROM split_strings
    WHERE remaining IS NOT NULL
)
SELECT id, TRIM(tag) as tag
FROM split_strings;
```
**Explanation**: This splits comma-separated tags into individual rows, like turning "red,blue,green" into three rows.

## Stored Procedures (SPs)

### What are Stored Procedures?
**Answer**: A stored procedure is a saved set of SQL commands you can run by calling its name, like a pre-written recipe you can reuse. It’s stored in the database and can handle complex tasks.

**Key features**:
- Can take input parameters, like ingredients for a recipe.
- Can return results, like a finished dish.
- Can perform multiple operations (INSERT, UPDATE, DELETE, SELECT).
- Stored in the database for easy access.

**Simple Example**:
```sql
-- Creating a stored procedure to get employee details
CREATE PROCEDURE GetEmployeeInfo
    @EmployeeID INT
AS
BEGIN
    SELECT Name, Department, Salary 
    FROM Employees 
    WHERE ID = @EmployeeID
END

-- Using the stored procedure
EXEC GetEmployeeInfo 101
```
**Explanation**: This stored procedure retrieves details for a specific employee when you provide their ID, like calling a pre-set query.

## User Defined Functions (UDFs)

### What are UDFs?
**Answer**: A User Defined Function is a custom function you create to calculate something specific, like a personalized calculator button you can use in queries.

**Key features**:
- Always returns a value, like a math result.
- Can take parameters, like numbers to add.
- Can be used in SELECT statements, like built-in functions.
- Cannot change database data (no INSERT/UPDATE/DELETE).

**Simple Example**:
```sql
-- Creating a function to calculate tax
CREATE FUNCTION CalculateTax(@Salary DECIMAL(10,2))
RETURNS DECIMAL(10,2)
AS
BEGIN
    DECLARE @Tax DECIMAL(10,2)
    SET @Tax = @Salary * 0.10  -- 10% tax
    RETURN @Tax
END

-- Using the function
SELECT Name, Salary, dbo.CalculateTax(Salary) AS Tax
FROM Employees
```
**Explanation**: This function calculates 10% tax on a salary and can be used in queries like any built-in function.

### Key Differences
- **Stored Procedures**: Handle multiple tasks (like a multi-step recipe), called with EXEC, can modify data, and return multiple results.
- **User Defined Functions**: Perform one specific calculation (like a calculator), used in queries, return one value, and cannot modify data.
**Think of SPs as "doing tasks" and UDFs as "calculating values"!**

## Quick Tips for SQL Interviews
- **Handle edge cases**: Consider NULLs, empty results, and duplicates in your queries.
- **Optimize queries**: Use indexes, avoid `SELECT *`, and write efficient JOINs.
- **Test your logic**: Walk through queries with sample data to ensure correctness.
- **Know your database**: Be aware of syntax differences (MySQL, PostgreSQL, SQL Server).
- **Master window functions**: They’re popular in interviews for advanced tasks.
- **Understand execution order**: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY.
- **Explain your approach**: Don’t just write code; clarify your reasoning clearly.
- **Practice with real data**: Get comfortable with SQL using realistic datasets.