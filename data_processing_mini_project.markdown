# Data Read and Process Mini Project

This document details a PySpark-based data processing mini project executed in a Databricks notebook. The project involves loading customer and order data from CSV files, performing data transformations, aggregations, and joins, and analyzing customer behavior. The code is organized into sections with detailed comments explaining each operation.

## Overview
The project processes two datasets:
- **Customers**: Contains customer information (e.g., `customer_id`, `city`, `state`, `country`, `registration_date`, `is_active`).
- **Orders**: Contains order details (e.g., `customer_id`, `order_date`, `total_amount`, `status`).

The pipeline includes:
- Loading and transforming data.
- Analyzing customer demographics and order patterns.
- Ranking customers by spend and order frequency.
- Saving results to Parquet files for further use.

## Prerequisites
- **Environment**: Databricks with PySpark.
- **Input Files**:
  - `customers.csv`: Customer data.
  - `orders.csv`: Order data.
- **Output Path**: `/Volumes/workspace/default/demo/outputfile` for saving results.

## Code with Comments

### 1. Initialize Spark Session
```python
# Import SparkSession from PySpark to create a Spark session
from pyspark.sql import SparkSession

# Create a Spark session with the application name "customerDataProcessing"
# getOrCreate() ensures a single session is reused if it already exists
spark = SparkSession.builder.appName("customerDataProcessing").getOrCreate()

# Display the Spark session object to confirm initialization
spark
```

### 2. Load Customer Data
```python
# Load customer data from a CSV file located in Databricks File System (DBFS)
# format("csv"): Specifies CSV format
# option("header", "true"): Treats the first row as column headers
# load(): Reads the file from the specified path
df = spark.read.format("csv").option("header", "true").load("/Volumes/workspace/default/demo/inputfiles/customers.csv")
```

### 3. Display Customer Data
```python
# Display the DataFrame in a tabular format in the Databricks notebook
# Useful for initial data inspection
display(df)
```

### 4. Transform Customer Data
```python
# Import all functions from pyspark.sql.functions for data transformations
from pyspark.sql.functions import *

# Transform the DataFrame:
# - Convert 'registration_date' to a proper date type using to_date()
# - Cast 'is_active' column to boolean type for consistency
df = df.withColumn('registration_data', to_date(col('registration_date'), 'yyyy-MM-dd')) \
       .withColumn('is_active', col('is_active').cast('boolean'))

# Display the transformed DataFrame to verify changes
df.display()
```

### 5. Count Unique Cities, States, and Countries
```python
# Calculate the number of unique cities, states, and countries
# select(countDistinct()): Counts distinct values in the specified column
# collect(): Retrieves the result as a list of rows
unique_cities = df.select(countDistinct('city')).collect()
unique_states = df.select(countDistinct('state')).collect()
unique_country = df.select(countDistinct('country')).collect()

# Print the counts of unique cities, states, and countries
# Access the first row and first column of the collected result
print(unique_cities[0][0])   # Number of unique cities
print(unique_states[0][0])  # Number of unique states
print(unique_country[0][0]) # Number of unique countries
```

### 6. Group and Count by Location
```python
# Group data by city, state, and country, count occurrences, and sort by count in descending order
# groupBy(): Groups data by the specified column
# count(): Counts rows in each group
# orderBy(col("count").desc()): Sorts by count in descending order
df.groupBy("city").count().orderBy(col("count").desc()).display()
df.groupBy("state").count().orderBy(col("count").desc()).display()
df.groupBy("country").count().orderBy(col("count").desc()).display()
```

### 7. Pivot by Active Status
```python
# Create pivot tables to show counts of active/inactive customers by city, state, and country
# groupBy(): Groups by the specified column
# pivot("is_active"): Creates columns for each value of 'is_active' (true/false)
# count(): Counts occurrences for each combination
df.groupBy("state").pivot("is_active").count().display()
df.groupBy("country").pivot("is_active").count().display()
df.groupBy("city").pivot("is_active").count().display()
```

### 8. Rank Customers by Registration Date within State
```python
# Import Window for ranking operations
from pyspark.sql.window import Window

# Define a window specification: partition by state, order by registration date descending
window_spec = Window.partitionBy("state").orderBy(col("registration_data").desc())

# Add ranking columns:
# - rank(): Assigns rank with gaps for ties
# - dense_rank(): Assigns rank without gaps for ties
# - row_number(): Assigns a unique sequential number
df.withColumn("rank", rank().over(window_spec)) \
  .withColumn("dense_rank", dense_rank().over(window_spec)) \
  .withColumn("row_number", row_number().over(window_spec)) \
  .display()
```

### 9. Filter Recent Customers
```python
# Filter customers registered after October 27, 2023
# lit(): Creates a literal value for comparison
# Note: The variable assignment is unnecessary as display() returns None
df_recent_customers = df.filter(col("registration_data") > lit("2023-10-27")).display()
# The line below is redundant since display() does not return a DataFrame
# df_recent_customers
```

### 10. Analyze Registration Dates
```python
# Group by city and find the oldest and newest registration dates
df.groupBy("city").agg(min("registration_data"), max("registration_data")).display()

# Count customers registered per day
df.groupBy("registration_data").count().display()

# Count customers registered per month using year and month functions
df.groupBy(year("registration_data"), month("registration_data")).count().display()
```

### 11. Save Customer Data to Parquet
```python
# Define output path for saving results
output_path = "/Volumes/workspace/default/demo/outputfile"

# Save the DataFrame as a Parquet file, overwriting if it exists
# mode("overwrite"): Overwrites existing files at the output path
df.write.mode("overwrite").parquet(output_path)

# List files in the output directory to verify
display(dbutils.fs.ls(output_path))
```

### 12. Load and Display Order Data
```python
# Load order data from a CSV file
# option("inferSchema", "true"): Automatically infers column data types
order_df = spark.read.format("csv").option("header", "true").option('inferSchema', 'true').load("/Volumes/workspace/default/demo/inputfiles/orders.csv")

# Display the order DataFrame
order_df.display()

# Redundant line: Simply referencing the DataFrame does nothing
# order_df
```

### 13. Join Customers and Orders
```python
# Perform an inner join between customers and orders on customer_id
# join(): Combines DataFrames based on the specified column and join type
customers_orders_df = df.join(order_df, 'customer_id', "inner")

# Display the joined DataFrame
customers_orders_df.display()

# Redundant line: groupBy without aggregation does nothing
# customers_orders_df.groupBy("customer_id")
```

### 14. Analyze Orders per Customer
```python
# Count orders per customer and sort by count in descending order
# This identifies customers with the highest order frequency
customers_order_count = customers_orders_df.groupBy("customer_id").count().orderBy(col("count").desc())
customers_order_count.display()
```

### 15. Calculate Total Spend per Customer
```python
# Calculate total spend per customer by summing total_amount
# Sort by total spend in descending order to find top-spending customers
customer_total_spend = customers_orders_df.groupBy("customer_id").agg(sum("total_amount")).orderBy(col("sum(total_amount)").desc())
customer_total_spend.display()
```

### 16. Calculate Average Spend per Customer
```python
# Calculate average spend per customer using avg()
# Sort by average spend in descending order
customer_average_spend = customers_orders_df.groupBy("customer_id").agg(avg("total_amount")).orderBy(col("avg(total_amount)").desc())
customer_average_spend.display()
```

### 17. Count Orders by Status
```python
# Count orders by status (e.g., completed, pending) and sort by count
order_status_count = customers_orders_df.groupBy("status").count().orderBy(col("count").desc())
order_status_count.display()
```

### 18. Analyze Orders by Month and Year
```python
# Count orders by month, extracted from order_date
# Sort by month for chronological order
order_by_month = customers_orders_df.groupBy(month("order_date").alias("month")).count().orderBy(col("month"))
order_by_month.display()

# Count orders by year, extracted from order_date
order_by_year = customers_orders_df.groupBy(year("order_date").alias("year")).count()
order_by_year.display()
```

### 19. Rank Customers by Total Spend
```python
# Define a window specification to rank customers by total spend
window_spec = Window.orderBy(col('sum(total_amount)').desc())

# Add a dense_rank column to rank customers by total spend
ranked_customers = customer_total_spend.withColumn('dense_rank', dense_rank().over(window_spec))
ranked_customers.display()
```

### 20. Compare Order Frequency and Total Spend
```python
# Join order count and total spend DataFrames to compare frequency vs. spend
# Sort by order count (descending) and total spend (ascending)
customer_spend_vs_orders = customers_order_count.join(
    customer_total_spend,
    'customer_id',
    "inner"
).orderBy(
    col("count").desc(),
    col("sum(total_amount)")
)
customer_spend_vs_orders.display()
```

### 21. Save Joined Data to Parquet
```python
# Save the joined customers and orders DataFrame as a Parquet file
# Overwrites existing files at the output path
output_path = "/Volumes/workspace/default/demo/outputfile"
customers_orders_df.write.mode("overwrite").parquet(output_path)

# List files in the output directory to verify
display(dbutils.fs.ls(output_path))
```

## Explanation of Key Operations

### Data Loading
- **Customers**: Loaded from `customers.csv` with columns like `customer_id`, `city`, `state`, `country`, `registration_date`, `is_active`.
- **Orders**: Loaded from `orders.csv` with columns like `customer_id`, `order_date`, `total_amount`, `status`.

### Transformations
- Converted `registration_date` to date type and `is_active` to boolean for consistency.
- Used window functions (`rank`, `dense_rank`, `row_number`) to rank customers by registration date within each state.

### Aggregations
- Counted unique cities, states, and countries.
- Grouped customers by location and active status.
- Analyzed registration dates (daily, monthly, min/max per city).

### Joins and Order Analysis
- Joined customers and orders on `customer_id` to analyze customer behavior.
- Calculated:
  - Total orders per customer.
  - Total and average spend per customer.
  - Order distribution by status, month, and year.
- Ranked customers by total spend and compared order frequency vs. spend to identify high-frequency, low-spending customers.

### Output
- Saved results as Parquet files for efficient storage and querying.
- Used `display()` for visualization in Databricks.

## Notes
- **Redundant Code**: Some lines (e.g., `df_recent_customers`, `customers_orders_df.groupBy("customer_id")`, `order_df` after `display()`) are unnecessary or have no effect.
- **Best Practices**:
  - Avoid redundant `display()` calls in production code.
  - Use meaningful column aliases in aggregations (e.g., `sum(total_amount).alias("total_spend")`).
  - Consider caching DataFrames (`df.cache()`) for repeated operations to improve performance.
- **Potential Improvements**:
  - Add error handling for file loading.
  - Validate input data schemas.
  - Optimize joins by broadcasting smaller DataFrames if applicable.

## Output Files
- Results are saved to `/Volumes/workspace/default/demo/outputfile` in Parquet format, suitable for further analysis or integration with other tools.