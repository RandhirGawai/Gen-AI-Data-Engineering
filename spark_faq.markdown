
![6A3E9071-34DB-4BD1-BCD8-2D4C6655221F_1_201_a](https://github.com/user-attachments/assets/148cc7ab-d986-4e0f-81d3-e9819cdb925e)
![C97FFC24-371C-4D58-93B2-024AF1E90E67_1_201_a](https://github.com/user-attachments/assets/2af2134c-7f60-4775-86d8-593ab3949de5)
![9E0F5879-3EE0-4F66-A125-7317BA614137_1_201_a](https://github.com/user-attachments/assets/e1598878-6914-4e3f-b4ff-63354a1eb7ff)
![6B3CAE3D-B783-4C06-9891-250D9E68FC1E_1_201_a](https://github.com/user-attachments/assets/78ee9c0d-2fb3-4254-89d2-c537673a8871)
![FFD1BBE8-0E97-49C1-B7EB-B905139E3DC5_1_201_a](https://github.com/user-attachments/assets/0f009065-d49d-40ea-9b00-01a36dc0e434)
![CA23A3FD-9850-40F8-ACA8-19E06E2CFBA9_1_201_a](https://github.com/user-attachments/assets/cd50bfa2-7d4e-448c-9685-3f483f311620)
![E22222C4-BA3E-46F5-A530-92110F59DDCA_1_201_a](https://github.com/user-attachments/assets/c16b9b03-28e5-496b-8ec7-61eefd3b524e)
![82BFF528-42B2-4EA9-93BC-3C29B5864BF5_1_201_a](https://github.com/user-attachments/assets/81553de1-ee36-4c27-8149-0ae708090684)
![B7F0F7A0-2464-4215-9ABA-02FD0BD512FC_1_201_a](https://github.com/user-attachments/assets/ed628a78-6afe-4175-896b-f7de1f9fdb68)
![7B7DDA8C-FB7C-4121-ACC4-509CF1456639_1_201_a](https://github.com/user-attachments/assets/93427f40-3ad3-4fb5-91e5-c53c6c48e5db)
![1](https://github.com/user-attachments/assets/68163978-287f-4be5-a5d1-297e96f68ef8)
![2](https://github.com/user-attachments/assets/e257e256-d17a-4061-a01d-e5325b3933fc)
![3](https://github.com/user-attachments/assets/f9d8655a-8c51-4607-885f-df0aff740e00)






# Apache Spark FAQ

## What are the different modes of execution in Apache Spark?
Spark can run in several modes depending on the deployment environment:

- **Local Mode**: Runs on a single machine, ideal for development and testing. Utilizes local CPU cores.
  - *Example*: Used to test a data pipeline with dummy sales data before cluster deployment.
- **Standalone Mode**: Spark manages its own cluster without external tools like Hadoop or YARN.
  - *Example*: Set up Spark on 3 VMs for a POC to process logs in standalone mode.
 ![5A639654-57D5-40E0-9C8E-82D159868BDF_1_201_a](https://github.com/user-attachments/assets/7689cfca-fd11-4100-9ca2-9fa80fc03014)
![734B6B99-6192-4F81-9FAA-3906C66FC525_1_201_a](https://github.com/user-attachments/assets/e0b3d685-5188-4db2-bf55-c3df5c1a37ab)


    
- **YARN Mode (Hadoop Cluster)**: Integrates with Hadoop’s YARN for resource management.
  - *Example*: Used in a retail project with an existing Hadoop cluster.
- **Mesos / Kubernetes Mode**: Runs on container-based systems like Kubernetes or Apache Mesos for dynamic resource allocation.
  - *Example*: Cloud-native setups use Kubernetes to auto-scale Spark jobs based on load.


# Spark and AWS Integration Guide

## 1️⃣ Connecting S3 Data in Databricks for PySpark

Databricks can directly read/write from Amazon S3 using the Hadoop S3A connector.

### Steps

#### A. Mount S3 to Databricks (Optional)
For a persistent path:

```python
dbutils.fs.mount(
  source = "s3a://<your-bucket-name>",
  mount_point = "/mnt/mydata",
  extra_configs = {"fs.s3a.access.key": "<ACCESS_KEY>",
                   "fs.s3a.secret.key": "<SECRET_KEY>"}
)
```

#### B. Read Directly Without Mounting

```python
df = spark.read.format("csv") \
    .option("header", "true") \
    .load("s3a://<bucket-name>/path/to/file.csv")
```

#### C. Write Back to S3

```python
df.write.format("parquet").mode("overwrite").save("s3a://<bucket-name>/output/")
```

### Note
- Configure AWS credentials via `spark.conf.set("fs.s3a.access.key", ...)` or use IAM roles if Databricks runs on AWS.

## 2️⃣ Using AWS Glue to Extract Retail Sales Data (Batch + Streaming)

AWS Glue is an ETL service for batch and streaming data from multiple sources.

### A. Sources
- **Batch**: S3, RDS, Redshift, DynamoDB, JDBC sources.
- **Streaming**: Kinesis Data Streams, Kafka, AWS IoT Core.

### B. Process

#### Batch Ingestion
- Create a Glue Crawler for batch sources (e.g., S3, JDBC) to create a Data Catalog table.
- Use Glue Spark jobs for transformation and loading.

#### Streaming Ingestion
- Create a Glue Streaming Job (uses Spark Structured Streaming).
- Read from Kinesis or Kafka.
- Apply PySpark transformations.
- Write to S3 in near real-time.

#### Example Glue Script for Streaming

```python
import sys
from awsglue.context import GlueContext
from pyspark.context import SparkContext
from awsglue.job import Job
from pyspark.sql.functions import col, when, lit, round, avg, sum as _sum, countDistinct
from pyspark.sql.types import DoubleType, IntegerType

# Initialize Glue job
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)

## -------------------------
## 1️⃣ READ MULTIPLE TABLES
## -------------------------
sales_df = glueContext.create_dynamic_frame.from_catalog(
    database="retail_db",
    table_name="sales_streaming_data",
    transformation_ctx="sales_df"
).toDF()

products_df = glueContext.create_dynamic_frame.from_catalog(
    database="retail_db",
    table_name="products_master",
    transformation_ctx="products_df"
).toDF()

customers_df = glueContext.create_dynamic_frame.from_catalog(
    database="retail_db",
    table_name="customers_master",
    transformation_ctx="customers_df"
).toDF()

## -------------------------
## 2️⃣ DATA CLEANING & TYPE CASTING
## -------------------------
# Remove duplicates
sales_df = sales_df.dropDuplicates()

# Handle nulls - replace null sales with 0
sales_df = sales_df.fillna({"sales": 0, "quantity": 0})

# Cast types
sales_df = sales_df \
    .withColumn("sales", col("sales").cast(DoubleType())) \
    .withColumn("quantity", col("quantity").cast(IntegerType()))

products_df = products_df.withColumn("price", col("price").cast(DoubleType()))

## -------------------------
## 3️⃣ JOIN TABLES
## -------------------------
# Join sales with product info
sales_products_df = sales_df.join(
    products_df,
    sales_df.product_id == products_df.product_id,
    "left"
)

# Join with customers
full_df = sales_products_df.join(
    customers_df,
    sales_products_df.customer_id == customers_df.customer_id,
    "left"
)

## -------------------------
## 4️⃣ FEATURE ENGINEERING
## -------------------------
# Add tax column
full_df = full_df.withColumn("sales_with_tax", round(col("sales") * 1.18, 2))

# Add discount flag
full_df = full_df.withColumn("discount_flag", when(col("discount") > 0, lit(1)).otherwise(lit(0)))

## -------------------------
## 5️⃣ FILTERING
## -------------------------
# Keep only transactions > 0 sales
full_df = full_df.filter(col("sales") > 0)

## -------------------------
## 6️⃣ AGGREGATIONS
## -------------------------
agg_df = full_df.groupBy("store_id") \
    .agg(
        _sum("sales").alias("total_sales"),
        avg("sales").alias("avg_sales"),
        countDistinct("customer_id").alias("unique_customers")
    )

## -------------------------
## 7️⃣ SORTING
## -------------------------
agg_df = agg_df.orderBy(col("total_sales").desc())

## -------------------------
## 8️⃣ WRITE STREAM TO S3
## -------------------------
agg_df.writeStream \
    .format("parquet") \
    .option("path", "s3://sales-processed/") \
    .option("checkpointLocation", "s3://sales-checkpoints/") \
    .outputMode("complete") \
    .start()

job.commit()

```

## 3️⃣ Transformation Architecture for PySpark

A typical PySpark pipeline includes:

1. **Ingestion Layer**: Read raw data from S3, databases, Kafka, etc.
2. **Staging Layer**: Store raw data in temporary S3 or Delta tables.
3. **Transformation Layer**:
   - Cleaning (null handling, type casting)
   - Filtering
   - Aggregations
   - Joins
   - Enrichments
4. **Storage Layer**: Write to curated S3 paths in Parquet/Delta format.
5. **Serving Layer**: BI tools (Power BI, QuickSight, Tableau) or ML pipelines consume curated data.

### Example Pipeline
```
Raw Data (S3/Kafka) → PySpark DataFrame → Transform (filter, join, agg) → Delta Table → BI Dashboard
```

## 4️⃣ RDD and Partitioning Techniques

### RDD (Resilient Distributed Dataset)
- **Definition**: Spark’s fundamental data structure, an immutable distributed collection of objects.
- DataFrames are built on RDDs, but DataFrames are preferred for most tasks.

### What is Partitioning in Spark?
- Determines how data is distributed across cluster nodes.
- Good partitioning reduces shuffling, improves parallelism, and boosts performance.
- Bad partitioning causes excessive shuffles and slow jobs.

### Why is Partitioning Important?
- Example: 1 billion records in 1 partition → processed by 1 executor (slow).
- 200 partitions → processed in parallel (faster).

### Partitioning Techniques

#### 1. Hash Partitioning
- Uses a hash function: `partition = hash(key) % num_partitions`.
- Same key goes to the same partition.
- Best for joins and `groupByKey`.

**Example**:

```python
rdd = sc.parallelize([
    (101, "Store_A"), 
    (102, "Store_B"), 
    (101, "Store_A"), 
    (103, "Store_C")
])

# Partition into 3 partitions based on store_id
partitioned_rdd = rdd.partitionBy(3)

print("Number of partitions:", partitioned_rdd.getNumPartitions())

def show_partitions(index, iterator):
    yield (index, list(iterator))

print(partitioned_rdd.mapPartitionsWithIndex(show_partitions).collect())
```

**Output**:
```
Number of partitions: 3
[(0, [(102, 'Store_B')]), 
 (1, [(101, 'Store_A'), (101, 'Store_A')]), 
 (2, [(103, 'Store_C')])]
```

#### 2. Range Partitioning
- Divides data into sorted key ranges (e.g., Jan–Mar, Apr–Jun).
- Use for sorted data or range queries.

**Example**:

```python
from pyspark.rdd import portable_hash
from pyspark import SparkContext

sc = SparkContext()

rdd = sc.parallelize([
    (5, "E"), (1, "A"), (3, "C"), (4, "D"), (2, "B")
])

def range_partitioner(key):
    if key <= 2:
        return 0
    elif key <= 4:
        return 1
    else:
        return 2

partitioned_rdd = rdd.partitionBy(3, range_partitioner)

print(partitioned_rdd.mapPartitionsWithIndex(show_partitions).collect())
```

**Output**:
```
[(0, [(1, 'A'), (2, 'B')]),
 (1, [(3, 'C'), (4, 'D')]),
 (2, [(5, 'E')])]
```

#### 3. Round Robin Partitioning (Default)
- Distributes records evenly without considering keys.
- Use when no key-based processing is needed.

**Example**:

```python
rdd = sc.parallelize(range(1, 11), numSlices=3)
print(rdd.mapPartitionsWithIndex(show_partitions).collect())
```

**Output**:
```
[(0, [1, 4, 7, 10]), 
 (1, [2, 5, 8]), 
 (2, [3, 6, 9])]
```

#### 4. Custom Partitioning
- Define your own partitioning logic.
- Use for specific data distribution needs.

**Example**:

```python
def custom_partitioner(customer_id):
    return 0 if customer_id < 500 else 1

rdd = sc.parallelize([
    (100, "Cust_A"), (200, "Cust_B"), (600, "Cust_C"), (700, "Cust_D")
])

partitioned_rdd = rdd.partitionBy(2, custom_partitioner)

print(partitioned_rdd.mapPartitionsWithIndex(show_partitions).collect())
```

**Output**:
```
[(0, [(100, 'Cust_A'), (200, 'Cust_B')]), 
 (1, [(600, 'Cust_C'), (700, 'Cust_D')])]
```

### Key Points
- **Default Partitioning**: `parallelize()` uses round robin.
- **Repartition vs Coalesce**:
  - `repartition(n)`: Increases/decreases partitions (shuffles).
  - `coalesce(n)`: Reduces partitions without full shuffle.
- **Partition Size**: Aim for 100–200 MB per partition for big data jobs.
- **Avoid Skew**: Ensure partitions are balanced to avoid performance bottlenecks.

## 5️⃣ Answers to Questions

### What are groupByKey and reduceByKey?

- **reduceByKey**: Aggregates values for a key using a reduce function. Combines locally first(local aggregation is there), reducing shuffle, this is fast(we can use sum, min, max function)

- **groupByKey**: Groups all values for a key into a list. Shuffles all data, memory-heavy, slower for aggregations, this is slow (we can use median, avg,mode function)

  ![D018C65B-222E-42D1-8334-1FD6106ED23D_1_201_a](https://github.com/user-attachments/assets/ad8bfa17-0735-409b-9a88-94de35e7ed6e)
  ![94C5C68C-634F-42D1-8D9C-1C4D7405B57C_1_201_a](https://github.com/user-attachments/assets/e5a2ca64-59f2-4426-8a68-343fd5a44baf)

- **reduceByKey**: is a transformation
- **reduceByValue**: is action( we can not perform further operation on it as it is not optimized)

#### groupByKey Example

```python
rdd = sc.parallelize([
    ("store1", 100), ("store2", 200), 
    ("store1", 150), ("store2", 250)
])

grouped = rdd.groupByKey().mapValues(list)
print(grouped.collect())
```

**Output**:
```
[('store1', [100, 150]), ('store2', [200, 250])]
```

**Performance Note**: `groupByKey` shuffles all data, so avoid for large aggregations.

#### reduceByKey Example

```python
rdd = sc.parallelize([
    ("store1", 100), ("store2", 200), 
    ("store1", 150), ("store2", 250)
])

reduced = rdd.reduceByKey(lambda x, y: x + y)
print(reduced.collect())
```

**Output**:
```
[('store1', 250), ('store2', 450)]
```

**Performance Note**: `reduceByKey` aggregates locally, reducing shuffle.

### Performance Comparison

| Feature          | groupByKey                     | reduceByKey                     |
|------------------|--------------------------------|---------------------------------|
| Purpose          | Group values into a list       | Aggregate values                |
| Shuffle Data     | All key-value pairs            | Only aggregated results          |
| Memory Usage     | High (stores all values)       | Lower                           |
| Speed            | Slower for aggregation         | Faster for aggregation          |
| When to Use      | Need all raw values            | Need aggregation                |

### Visual Example

**Data**:
```
Partition 1: ("A", 2), ("B", 3)
Partition 2: ("A", 5), ("B", 7)
```

**groupByKey**:
```
Shuffle → ("A", [2, 5]), ("B", [3, 7])
(All values move across network)
```

**reduceByKey (sum)**:
```
Local reduce:
  Partition 1: ("A", 2), ("B", 3)
  Partition 2: ("A", 5), ("B", 7)
Shuffle reduced values:
  ("A", 7), ("B", 10)
```

### Key Interview Points
- Prefer `reduceByKey` for aggregations to minimize shuffle.
- Use `countByKey()` or `mapValues(lambda x: 1).reduceByKey(lambda a,b: a+b)` for counting instead of `groupByKey`.
- `aggregateByKey` offers flexible custom aggregation.

## 6️⃣ PySpark Basics & Core Concepts

### 1.1 RDDs vs DataFrames vs Datasets
![1D3119D1-22EF-4B87-83BA-6013E105BD5A_4_5005_c](https://github.com/user-attachments/assets/59c93f67-af00-4046-ad97-34da3d998b6a)


| Feature          | RDD                                    | DataFrame                              | Dataset (Scala/Java only)              |
|------------------|----------------------------------------|----------------------------------------|----------------------------------------|
| Definition       | Low-level distributed collection       | Distributed table with named columns   | Strongly typed DataFrame               |
| Type Safety      | No                                     | No                                     | Yes                                    |
| Performance      | Lower (no optimization)                | Higher (Catalyst optimizer)            | Higher (Catalyst + type safety)        |
| Ease of Use      | Complex (manual functions)             | Easier (SQL, high-level API)           | Medium                                 |
| When to Use      | Custom processing, unstructured data    | Structured/semi-structured data        | Type-safe structured data              |

**Example**:

```python
# RDD example
rdd = sc.parallelize([1, 2, 3, 4])
rdd2 = rdd.map(lambda x: x * 2)
print(rdd2.collect())

# DataFrame example
df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "name"])
df.show()
```
![FA2CD637-F5C1-4CF4-AF05-6F2CF27FA58E_1_201_a](https://github.com/user-attachments/assets/52e6733c-32d0-4e4c-9e09-0866a0db229c)
![55B84F1E-D042-4498-BFDB-DE92F756E2DB_1_201_a](https://github.com/user-attachments/assets/3a60dbef-5497-48a6-a430-e71e4e6941e2)


**Rule of Thumb**:
- Use DataFrames for most tasks (optimized by Spark).
- Use RDDs for low-level control.

# Spark Data Structures: RDD, DataFrame, and Dataset

## RDD (Resilient Distributed Dataset)

The main abstraction Spark provides is a **Resilient Distributed Dataset (RDD)**, a collection of elements partitioned across the nodes of a cluster that can be operated on in parallel.

### RDD Features

- **Distributed Collection**:  
  RDDs use MapReduce operations for processing and generating large datasets with parallel, distributed algorithms. Users can write parallel computations using high-level operators without worrying about work distribution or fault tolerance.

- **Immutable**:  
  RDDs consist of partitioned records. A partition is the basic unit of parallelism, and each partition is an immutable logical division of data created through transformations. Immutability ensures consistency in computations.

- **Fault Tolerant**:  
  If a partition is lost, Spark can recompute it using the lineage of transformations, avoiding data replication across nodes. This is a key benefit, saving effort in data management and enabling faster computations.

- **Lazy Evaluations**:  
  Transformations in Spark are lazy, meaning they do not compute results immediately. They record transformations applied to a base dataset, and computation occurs only when an action (e.g., `collect()`, `count()`) is triggered.
  Here it is lazy because we dont want to unnecesarily utilize our resource for transformation until action get called.

- **Functional Transformations**:  
  RDDs support:
  - **Transformations**: Create a new dataset (e.g., `map()`, `filter()`).
  - **Actions**: Return a value to the driver program after computation (e.g., `reduce()`, `collect()`).

- **Data Processing Formats**:  
  RDDs efficiently process both structured and unstructured data.

- **Programming Languages Supported**:  
  RDD API is available in Java, Scala, Python, and R.

### RDD Limitations

- **No Inbuilt Optimization Engine**:  
  RDDs lack Spark’s advanced optimizers (Catalyst Optimizer, Tungsten Execution Engine) when working with structured data, requiring developers to manually optimize based on RDD attributes.

- **Handling Structured Data**:  
  RDDs do not infer the schema of data, requiring users to specify it manually, unlike DataFrames and Datasets.

## DataFrame
![6CE00AE4-C6CC-4CA1-8064-D35382052A78_1_201_a](https://github.com/user-attachments/assets/d1be1c3b-b8c5-4eed-974a-306d2a406510)
![55B84F1E-D042-4498-BFDB-DE92F756E2DB_1_201_a](https://github.com/user-attachments/assets/fdf6e329-6170-4293-a62e-3b551a97708a)


Introduced in Spark 1.3, **DataFrames** overcome key challenges of RDDs. A DataFrame is a distributed collection of data organized into named columns, conceptually equivalent to a table in a relational database or a DataFrame in R/Python. Spark also introduced the Catalyst Optimizer for query optimization.

- **Distributed collection of Row Object**: A DataFrame is a distributed collection of data organized into named columns. It is conceptually equivalent to a table in a relational database, but with richer optimizations under the hood.

- **Data Processing**: Processing structured and unstructured data formats (Avro, CSV, elastic search, and Cassandra) and storage systems (HDFS, HIVE tables, MySQL, etc). It can read and write from all these various datasources.

- **Optimization using catalyst optimizer**: It powers both SQL queries and the DataFrame API. Dataframe use catalyst tree transformation framework in four phases,



### DataFrame Limitations

- **Compile-Time Type Safety**:  
  DataFrame APIs lack compile-time safety, limiting manipulation when the data structure is unknown. For example:

  ```scala
  case class Person(name: String, age: Int)
  val dataframe = sqlContext.read.json("people.json")
  dataframe.filter("salary > 10000").show
  // Throws Runtime Exception: cannot resolve 'salary' given input age, name
  ```

  This is challenging for complex transformation and aggregation pipelines.

- **Cannot Operate on Domain Object**:  
  Once a domain object is transformed into a DataFrame, the original RDD cannot be regenerated. For example:

  ```scala
  case class Person(name: String, age: Int)
  val personRDD = sc.makeRDD(Seq(Person("A", 10), Person("B", 20)))
  val personDF = sqlContext.createDataFrame(personRDD)
  personDF.rdd // Returns RDD[Row], not RDD[Person]
  ```

## Dataset API

The **Dataset API**, an extension to DataFrames, provides a type-safe, object-oriented programming interface. It is a strongly-typed, immutable collection of objects mapped to a relational schema.

### Dataset Features

- **Best of RDD and DataFrame**:  
  Combines RDD’s functional programming and type safety with DataFrame’s relational model, query optimization, Tungsten execution, sorting, and shuffling.

- **Encoders**:  
  Encoders convert JVM objects into Datasets, enabling work with structured and unstructured data. Spark 1.6 supports automatic encoder generation for primitive types (e.g., String, Integer), Scala case classes, and Java Beans.

- **Programming Languages Supported**:  
  Available in Scala and Java (Python and R support added in later versions, e.g., Spark 2.0 for Python).

- **Type Safety**:  
  Datasets provide compile-time safety, allowing operations on domain objects with lambda functions. For example:

  ```scala
  case class Person(name: String, age: Int)
  val personRDD = sc.makeRDD(Seq(Person("A", 10), Person("B", 20)))
  val personDF = sqlContext.createDataFrame(personRDD)
  val ds: Dataset[Person] = personDF.as[Person]
  ds.filter(p => p.age > 25) // Works
  ds.filter(p => p.salary > 25) // Compile-time error: 'salary' not a member of Person
  ds.rdd // Returns RDD[Person]
  ```

- **Interoperable**:  
  Easily converts RDDs and DataFrames into Datasets without boilerplate code.

### Dataset API Limitation

- **Requires Type Casting to String**:  
  Querying Datasets requires specifying fields as strings, and results must be cast to the required data type. Map operations on Datasets do not use the Catalyst Optimizer. For example:

  ```scala
  ds.select(col("name").as[String], $"age".as[Int]).collect()
  ```


### 1.2 Lazy Evaluation
- Spark builds a logical execution plan (DAG) and delays execution until an action is called (e.g., `collect()`, `count()`, `show()`).

**Example**:

```python
rdd = sc.parallelize([1, 2, 3])
mapped = rdd.map(lambda x: x * 2)  # No execution
filtered = mapped.filter(lambda x: x > 2)  # No execution
print(filtered.collect())  # Execution happens
```

**Why Important**: Spark optimizes the DAG for better performance.

### 1.3 Transformations vs Actions
- **Transformation**: After every operation a new RDD/DataFrame created (lazy). Examples: `map()`, `filter()`, `groupByKey()`.
- Represented as an intermediate node in our DAG
- **Action**: An operations that execute the Triggers and returns results to driver, or writes output. Examples: `count()`, `collect()`, `saveAsTextFile()`,`first()`.
- Represented as a terminal node.
![5F4850C8-4E6A-4EA1-B6FE-9351EE18403C_1_201_a](https://github.com/user-attachments/assets/e212a0d9-2cf5-4f99-89cc-011e7108bb21)

### 1.4 Narrow vs Wide Transformations
- **Narrow**: No shuffle (data stays in partition). Examples: `map()`, `filter()`.
              Each child partition depend on single parent partition
              It is faster as it dont required shuffling
              Example: suppose Pune is appear 100 time in table so instead of sending Pune 100 time it will return Pune, 100
- **Wide**: Requires shuffle across partition( requiring network i/o). Examples: `reduceByKey()`, `join()`.
              Each child is depent on multiple parent partition
              It is slower as it required shuffling
  ![B85E5EA9-5183-419F-A7EB-B352FA61669D_1_201_a](https://github.com/user-attachments/assets/1feda057-ad69-4910-9506-322e192070ad)


            

**Example**:

```python
# Narrow transformation
rdd.map(lambda x: x * 2)

# Wide transformation
rdd.reduceByKey(lambda a, b: a + b)
```

### 1.5 SparkSession & Context
- **SparkSession**: Entry point for Spark functionality.
- **SparkContext**: Handles RDD operations.
- **SQLContext**: Enables SQL queries (now part of SparkSession).

**Example**:

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Example").getOrCreate()
sc = spark.sparkContext
```

## 7️⃣ Performance Optimization

### 2.1 Partitioning
- `repartition(n)`: Increases/decreases partitions (shuffles).
- `coalesce(n)`: Reduces partitions without shuffle.
- Custom partitioning for key-based joins.

**Example**:

```python
df = df.repartition(10, "customer_id")
df = df.coalesce(5)
```

### 2.2 Caching/Persisting
- `cache()` = `persist(StorageLevel.MEMORY_ONLY)`.
- `persist(MEMORY_AND_DISK)`: Stores in memory, spills to disk if needed.
- Use when reusing DataFrames multiple times.

### 2.3 Broadcast Variables
- Send small datasets to all executors to avoid shuffles.

**Example**:

```python
small_df = spark.read.csv("lookup.csv")
broadcast_lookup = sc.broadcast(small_df.collect())
```

### 2.4 Shuffle Optimization
- Shuffles occur in joins, `groupByKey`, `distinct`, `repartition`.
- Reduce shuffles by:
  - Broadcasting small datasets.
  - Using `reduceByKey` instead of `groupByKey`.

### 2.5 File Formats
- **Parquet/ORC**: Columnar, compressed, reads only needed columns.
- **CSV**: Row-based, larger, slower for big data.

## 8️⃣ Advanced Transformations

### 3.1 map(), flatMap(), mapPartitions()

```python
rdd.map(lambda x: [x, x*2])        # [[1, 2], [2, 4]]
rdd.flatMap(lambda x: [x, x*2])    # [1, 2, 2, 4]
```

### 3.2 groupByKey() vs reduceByKey() vs aggregateByKey()
- `groupByKey()`: Groups into lists (expensive).
- `reduceByKey()`: Aggregates locally then globally.
- `aggregateByKey()`: Custom aggregation.

### 3.3 distinct(), union(), intersection()

```python
rdd.distinct()
rdd1.union(rdd2)
rdd1.intersection(rdd2)
```

### 3.4 Window Functions

```python
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank

windowSpec = Window.partitionBy("dept").orderBy("salary")
df.withColumn("rank", rank().over(windowSpec))
```

## 9️⃣ Common Coding Exercises

### 4.1 Word Count

```python
text = sc.textFile("file.txt")
counts = text.flatMap(lambda x: x.split(" ")) \
             .map(lambda x: (x, 1)) \
             .reduceByKey(lambda a, b: a + b)
```

### 4.2 Most Frequent Value

```python
df.groupBy("col").count().orderBy(desc("count"))
```

### 4.3 Top N per Group

```python
from pyspark.sql.window import Window
windowSpec = Window.partitionBy("dept").orderBy(desc("salary"))
df.withColumn("rank", row_number().over(windowSpec)).filter("rank <= 3")
```

### 4.4 Join & Aggregate

```python
df1.join(df2, "id").groupBy("dept").agg(sum("salary"))
```

### 4.5 Remove Duplicates

```python
df.dropDuplicates(["col"])
```

### 4.6 Custom Aggregation

```python
rdd.aggregateByKey(0, lambda a, b: a + b, lambda a, b: a + b)
```

### 4.7 Explode JSON

```python
from pyspark.sql.functions import from_json, explode
schema = ...
df = df.withColumn("json_col", from_json("json_string", schema))
df = df.withColumn("exploded", explode("json_col.array_field"))
```

## 10️⃣ Spark Architecture

### Concept
- **Driver Program**:
  Creating DAG from user defined code
  Scheduling tasks and managing the execution plan
  collect the result from executor
  Runs `main()`, converts transformations to a logical plan, optimizes, and assigns tasks to executors.
- **Spark Context**:
  Entry point for the spark functionality
- **Executors**:
  Run on worker nodes.
  Execute tasks assign by driver
  Store result in memory/disk.
- **Cluster Manager**:
  Manage & allocates resources (e.g., Standalone, YARN, Mesos, Kubernetes).
- **Task**:
  Smallest unit of work in spark created by spiltting data

- **DAG**: Direct A cyclic Graph
  Convert high level transformation(map, filter) to series of stages
  Stage can have task to execute 
  


### Common Interview Questions
- **What is the role of the driver in Spark?**  
  Converts code to a logical plan, optimizes it, and schedules tasks.
- **What happens if an executor fails?**  
  Spark recomputes lost partitions using RDD lineage.
- **Can we run Spark without YARN?**  
  Yes, using Standalone, Mesos, or Kubernetes.
- **How does Spark know how many executors to use?**  
  Configured via cluster manager settings (e.g., `--num-executors` in YARN).

## 11️⃣ Additional Concepts and Questions

### 5.2 Narrow vs Wide Transformations
- **Narrow**: No shuffle. Examples: `map()`, `filter()`.
- **Wide**: Requires shuffle. Examples: `groupByKey()`, `reduceByKey()`, `join()`.

**Questions**:
- **Which is faster?** Narrow (no shuffle).
- **What triggers a shuffle?** Operations like `groupByKey`, `join`, `repartition`.
- **How to reduce shuffle cost?** Use `reduceByKey`, broadcast joins, or partition wisely.

### 5.3 persist() vs cache()
- `cache()` = `persist(StorageLevel.MEMORY_ONLY)`.
- `persist()`: Offers options like `MEMORY_AND_DISK`, `DISK_ONLY`, `MEMORY_ONLY_SER`.

**Questions**:
- **If dataset is too big for memory, what happens with cache()?**  
  It fails or spills to disk if memory is insufficient.
- **When to use persist(MEMORY_AND_DISK)?**  
  When data is reused and memory is limited.
- **Does cache() survive executor failure?**  
  No, recomputes from lineage.

### 5.4 Handling Small Lookup Tables (Broadcast Join)

```python
from pyspark.sql.functions import broadcast
df1.join(broadcast(df2), "id")
```

# Data Orchestration for PySpark Workflows

Orchestration involves coordinating and automating the execution of multiple data processing tasks in the correct order, at the right time, with proper dependency management. Think of it like a film director: the actors (ETL jobs, PySpark scripts, SQL queries, machine learning pipelines) know their roles, but the director ensures each scene happens in sequence, connecting everything for a smooth flow.

## Why Orchestration is Needed

In real-world projects, you rarely deal with a single PySpark script. A typical pipeline might include:

- **Extract**: Pull data from APIs, databases, or streaming sources.
- **Transform**: Clean, join, and aggregate using PySpark.
- **Load**: Save to S3, Snowflake, or a data warehouse.
- **Post-processing**: Notify teams, trigger dashboards, or train ML models.

Without orchestration, you’d need to manually run each step. If a step fails, determining what to rerun becomes complex and error-prone.

## What Orchestration Does

- **Scheduling**: Run jobs at specific times (e.g., 2 AM daily) or intervals (e.g., every 15 minutes for streaming).
- **Dependency Management**: Ensures Step 2 runs only after Step 1 completes successfully.
- **Retry Logic**: Automatically retries failed steps.
- **Parallel Execution**: Runs independent jobs simultaneously to save time.
- **Monitoring**: Tracks job status, logs, and failures for debugging and auditing.

## Orchestration Tools for PySpark

| Tool                  | Where Used        | Best For                                     |
|-----------------------|-------------------|----------------------------------------------|
| Apache Airflow        | Cloud/On-prem     | Complex batch workflows, scheduling          |
| AWS Step Functions    | AWS               | Serverless orchestration for Glue, EMR, Lambda |
| Databricks Workflows  | Databricks        | Orchestrating notebooks & PySpark jobs       |
| Azure Data Factory    | Azure             | GUI-based orchestration for cloud pipelines  |
| Luigi                 | Python-based      | Simple DAG-based workflows                   |
| Prefect               | Hybrid            | Pythonic orchestration with minimal setup    |

## Example: Orchestrating a PySpark Retail ETL Pipeline

1. **Step 1**: Ingest sales data from MySQL (batch) → PySpark job.
2. **Step 2**: Ingest live POS transactions from Kafka (streaming) → PySpark Structured Streaming.
3. **Step 3**: Join batch and streaming datasets → write to Delta Lake in S3.
4. **Step 4**: Trigger BI dashboard refresh.
5. **Step 5**: Send Slack notification when the job completes.

An orchestration tool ensures these steps execute in the correct order, with proper error handling and retries.



**Questions**:
- **How does Spark decide when to broadcast?**  
  Automatically if table size < `spark.sql.autoBroadcastJoinThreshold` (default: 10MB).
- **What’s the default broadcast threshold?**  
  10MB, configurable.

### 5.5 Why Parquet over CSV
- **Parquet**: Columnar, compressed, supports projection pushdown.
- **CSV**: Row-based, larger, slower.

**Questions**:
- **Why is Parquet better for analytical queries?**  
  Reads only needed columns, compressed, faster.
- **Which format for archival?**  
  Parquet/ORC for compression and schema; CSV/JSON for readability.

### 5.6 Fault Tolerance
- Spark uses RDD lineage to recompute lost partitions.
- Checkpointing saves data to reliable storage (e.g., S3) to cut lineage.

**Questions**:
- **How is Spark fault-tolerant?**  
  Recomputes lost partitions via lineage.
- **RDD lineage vs checkpointing?**  
  Lineage tracks transformations; checkpointing saves data, breaking lineage.

### 5.7 What Happens on Action
- Transformations are lazy; actions trigger execution.
- Logical plan → optimized physical plan → DAG Scheduler → tasks to executors.

**Questions**:
- **What is lazy evaluation?**  
  Delays execution until an action is called.
- **Example of delayed execution?**  
  `df.filter(...).groupBy(...).count()` executes only on `.show()`.

### 5.8 repartition() vs coalesce()
- `repartition(n)`: Increases/decreases partitions, shuffles.
- `coalesce(n)`: Reduces partitions, no shuffle.

**Questions**:
- **Which is better for reducing partitions?**  
  `coalesce` (less costly).
- **Why is repartition slower?**  
  Involves full shuffle.

### 5.9 Checkpointing
- Saves RDD/DataFrame to reliable storage, cuts lineage.
- Useful for long-running jobs to avoid recomputation.

**Questions**:
- **Caching vs checkpointing?**  
  Caching stores in memory/disk; checkpointing saves to storage, breaks lineage.
- **When to use checkpointing?**  
  For long lineage or iterative jobs.

### 5.10 Spark Streaming vs Structured Streaming
- **Spark Streaming**: Uses DStreams, micro-batch processing.
- **Structured Streaming**: DataFrame/Dataset API, event-time processing, better optimizations.

**Questions**:
- **Which is better for exactly-once processing?**  
  Structured Streaming.
- **Can Structured Streaming process real-time without micro-batch?**  
  Yes, in continuous processing mode.

### Read data from HDFS

![88A54F83-30D4-477A-AA79-9A082C8DE68E_1_201_a](https://github.com/user-attachments/assets/c33c9731-a726-4860-9899-ac290d8d8938)
![89725920-4C30-44CF-A480-3AF94F74AE1E_1_201_a](https://github.com/user-attachments/assets/3308349d-68c5-464c-8e4d-9006672e5223)
![A35CFE3D-3158-4E74-9E85-0CF7AA9162DC_1_201_a](https://github.com/user-attachments/assets/d2caf67a-bfc9-42c9-9386-543e24aca5cd)
![7E428FEC-09C7-43BE-9D6E-9E76D815B519_1_201_a](https://github.com/user-attachments/assets/70611770-5315-4f87-8184-7a4329c03501)
![1BC2854E-32CD-451D-AA70-B08712793578_1_201_a](https://github.com/user-attachments/assets/feb7440c-e994-454d-8752-6b86c5402b57)





# Spark Schema Enforcement and I/O Modes

## Schema Enforcement in Spark DataFrame

**Schema enforcement** ensures that data being read or written matches a predefined schema (column names, data types, and nullability) instead of relying on Spark's automatic schema inference.

### Purpose
- Prevents incorrect data types.
- Ensures consistent data processing across environments.
- Avoids runtime errors due to unexpected schema changes.

### Example

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SchemaEnforcement").getOrCreate()

# Define schema
schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True)
])

# Read CSV with enforced schema
df = spark.read.csv("/path/to/data.csv", schema=schema, header=True)
df.printSchema()
```

**Behavior**:
- Spark fails if "age" contains non-integer values.
- Columns are strictly treated as per defined types.

## Schema Inference vs Schema Enforcement

| **Aspect**                | **Schema Inference**                              | **Schema Enforcement**                          |
|---------------------------|--------------------------------------------------|------------------------------------------------|
| **Definition**            | Spark guesses data types from sample rows.       | You define the schema manually.                |
| **Risk**                  | Wrong type detection (e.g., "1000" as String).   | Guaranteed correct types.                      |
| **Performance**           | Slower (requires data scan).                    | Faster (schema predefined).                    |

## Challenges in Schema Inference

When using `inferSchema=True`, the following issues may arise:

### a) Incorrect Inference
- Spark infers types from sampled data (default: partial scan).
- Misleading early rows can lead to incorrect schema (e.g., "1000" as Integer, but later rows contain "N/A").

### b) Production Issues
- **Schema Drift**: Incoming files in production may have:
  - Additional columns.
  - Missing columns.
  - Changed column order.
- This can cause pipeline failures or incorrect results.

### c) Performance Issues
- Schema inference requires two scans:
  1. Detect schema.
  2. Load data.
- This slows ingestion for large datasets.

## Best Practices
- Always define schema explicitly for production pipelines.
- Use `StructType` and `StructField` for strict enforcement.
- Validate schema at ingestion to detect drift early.
- Use schema evolution strategies (e.g., Delta Lake, Iceberg) for evolving data sources.

## Interview Q&A

**Q1: What is schema enforcement in Spark?**  
A: Applying a predefined schema to DataFrames/Datasets to validate column names, types, and nullability rules.

**Q2: What are the disadvantages of relying on `inferSchema=True`?**  
A:
- Incorrect type detection due to misleading sample rows.
- Extra job time from double scanning.
- Risk of schema drift in production causing failures.

**Q3: How do you enforce schema in Spark when reading a file?**  
A: Pass a `StructType` schema to the read method, e.g.:

```python
spark.read.csv(path, schema=mySchema, header=True)
```

**Q4: Give an example of a production issue caused by schema inference.**  
A: A sales CSV had "discount" as integers for months, but one month included "N/A". Spark inferred Integer, causing a runtime failure when "N/A" appeared.

**Q5: How does schema inference impact performance?**  
A: It requires an extra scan to detect the schema, slowing ingestion for large datasets.

## Read Modes in Spark

When reading data (e.g., CSV, JSON) into a DataFrame, Spark handles corrupt or bad records based on the `mode` option.

| **Mode**         | **Behavior**                                                                 | **Example**                                          |
|-------------------|-----------------------------------------------------------------------------|------------------------------------------------------|
| **permissive** (default) | Keeps all rows; bad records go to `_corrupt_record` column; missing values set to null. | `spark.read.option("mode", "PERMISSIVE").csv("file.csv")` |
| **dropMalformed** | Drops rows with corrupt records.                                             | `spark.read.option("mode", "DROPMALFORMED").csv("file.csv")` |
| **failFast**      | Throws an exception immediately on bad records.                              | `spark.read.option("mode", "FAILFAST").csv("file.csv")` |

**💡 Interview Tip**: `failFast` ensures data quality by stopping processing on invalid data, ideal for production pipelines where bad data must be addressed immediately.

## Is Write Operation Action or Transformation?

A write operation (e.g., `.write.csv(...)`, `.write.parquet(...)`) is an **Action**, not a transformation.

- **Why?** It triggers Spark to execute the DAG and output data.
- Transformations (e.g., `filter`, `select`) are lazy, building a logical plan, while write forces execution.

### Example

```python
# Transformation (lazy)
df_filtered = df.filter(df.age > 30)

# Action (triggers execution)
df_filtered.write.mode("overwrite").parquet("/data/output")
```

## Write Modes in Spark

Write modes define behavior when the output path or table already exists:

| **Mode**                  | **Behavior**                                     |
|---------------------------|-------------------------------------------------|
| **overwrite**             | Deletes existing data and writes new data.       |
| **append**                | Adds new data to existing data.                 |
| **ignore**                | Skips writing if data already exists (no error). |
| **error / errorifexists** (default) | Throws an error if data already exists.          |

### Example

```python
df.write.mode("overwrite").parquet("/data/output")
```

## Options in Write Operation

- `mode`: overwrite, append, ignore, error.
- `format`: parquet, csv, json, jdbc, orc.
- `partitionBy`: Creates folders for each partition column.
- `bucketBy`: Organizes data into buckets (requires Hive support).
- `option`: Configs like compression, delimiter, etc.

### Example

```python
df.write \
  .mode("append") \
  .format("parquet") \
  .option("compression", "snappy") \
  .partitionBy("country", "year") \
  .save("/data/output")
```

## Possible Interview Questions

**Q1: What are Spark read modes and when would you use `failFast`?**  
A1: Read modes handle malformed records. `failFast` stops processing on bad data, ensuring strict data quality in production.

**Q2: Is `.write()` in Spark a transformation or an action? Why?**  
A2: It’s an action because it triggers job execution and writes data to the sink.

**Q3: What’s the difference between `overwrite` and `append` in write mode?**  
A3: `overwrite` replaces existing data; `append` adds to it.

**Q4: How would you prevent overwriting existing data accidentally?**  
A4: Use `.mode("error")` (default) or `.mode("ignore")`.

**Q5: What’s the difference between `partitionBy` and `bucketBy` in writes?**  
A5: `partitionBy` creates folders by column values; `bucketBy` organizes data into a fixed number of buckets for joins/aggregations, requiring Hive support.



# Spark Job Execution and Deployment Modes

## Spark Job Execution Flow

A typical Spark job execution involves multiple stages, from writing code to completion on the cluster. Below is a step-by-step breakdown:

### 1. Writing Spark Code
Spark code defines transformations and actions. For example:

```python
df = spark.read.csv("data.csv", header=True, inferSchema=True)
result = df.groupBy("category").count()
result.show()
```

When executed, Spark builds a logical plan rather than processing data immediately.

### 2. Driver Program
The **driver** is the central component of a Spark application, running in the JVM where `SparkContext` or `SparkSession` is created.

**Responsibilities**:
- Parse code and create a logical plan (Directed Acyclic Graph, DAG).
- Optimize the logical plan using the Catalyst Optimizer.
- Convert the logical plan into a physical plan.
- Split the physical plan into stages and tasks.

### 3. Cluster Manager
The cluster manager allocates resources for Spark applications. Options include:
- Standalone
- YARN
- Mesos
- Kubernetes

The driver contacts the cluster manager to request executors.

### 4. Executors
Executors are JVM processes on worker nodes.

**Responsibilities**:
- Execute tasks assigned by the driver.
- Store data in memory/disk for caching and shuffle.
- Report results or status to the driver.

### 5. Stages and Tasks
- **Stage**: A set of tasks that can be executed in parallel, created at shuffle boundaries.
- **Task**: The smallest unit of work, processing a single partition of data.

### 6. Job Execution Flow
For the example `df.groupBy().count()`:
1. **Read Data**: Tasks read partitions from HDFS/S3/local.
2. **Map Stage**: Executors apply transformation logic to partitions.
3. **Shuffle**: Data is redistributed for grouping.
4. **Reduce Stage**: Grouped data is aggregated.
5. **Action (show)**: Results are collected to the driver.

### 7. Execution Flow Diagram

```
Driver Program
    |
    |---> Logical Plan (Unoptimized)
    |---> Optimized Logical Plan
    |---> Physical Plan (Stages)
    |
Cluster Manager
    |
    |---> Executors (on worker nodes)
           |---> Tasks run on partitions
           |---> Store cache if required
```

**Key Takeaway**:
- **Driver**: Plans the work.
- **Executors**: Perform the work.
- **Cluster Manager**: Allocates resources.

## Hadoop YARN Components

In Hadoop YARN (Yet Another Resource Negotiator), components are organized into three layers: Resource Management, Node Management, and Application Management.

### 1. Resource Manager (RM) – Master Component
Runs on the master node, managing cluster-wide resources and scheduling jobs.

**Sub-components**:
- **Scheduler**:
  - Allocates resources (CPU, memory) to applications based on policies.
  - Does not monitor job execution or retry failed tasks.
- **Application Manager**:
  - Manages application lifecycles.
  - Handles submissions, negotiates containers for the Application Master, and restarts it on failure.

### 2. Node Manager (NM) – Slave Component
Runs on each worker node.

**Responsibilities**:
- Manages node resources (CPU, memory, disk, network).
- Launches and monitors containers assigned by the Resource Manager.
- Reports node health and container status to the RM.

### 3. Application Master (AM)
Per-application process (e.g., one per Spark job).

**Responsibilities**:
- Negotiates resources from the Resource Manager.
- Works with Node Managers to execute tasks in containers.
- Handles application-specific fault tolerance and task scheduling.

### 4. Containers
Execution environments for tasks, allocated specific resources (memory, vCores), and managed by the Node Manager.

### Conceptual Flow Diagram

```
+--------------------+
| Resource Manager   |  <-- Cluster-wide master
|  - Scheduler       |
|  - App Manager     |
+--------------------+
        ↑
        |  (allocates containers)
        ↓
+--------------------+     +--------------------+
| Node Manager (NM)  |     | Node Manager (NM)  |   <-- Worker nodes
|  - Containers      |     |  - Containers      |
+--------------------+     +--------------------+
```

## Spark Deployment Modes

Spark deployment modes determine where the driver program runs relative to the cluster.

### 1. Local Mode
- **Description**: Runs Spark entirely on a single machine (driver + executors in one JVM).
- **When Used**: Development, testing, debugging small datasets.
- **Example Command**:

```bash
spark-submit --master local[4] my_app.py
```

`(local[4] → use 4 CPU threads)`

### 2. Cluster Mode
- **Description**: Driver runs on a worker node within the cluster, managed by the cluster manager (YARN, Mesos, Kubernetes, or Standalone).
- **Advantages**:
  - Ideal for production.
  - Driver proximity to executors reduces network latency.
- **Example Command (YARN Cluster)**:

```bash
spark-submit --master yarn --deploy-mode cluster my_app.py
```

- **Flow**:
  1. Submit application to cluster manager.
  2. Cluster manager launches driver in the cluster.
  3. Driver requests executors and coordinates the job.

### 3. Client Mode
- **Description**: Driver runs on the machine where `spark-submit` is executed; executors run on the cluster.
- **Advantages**:
  - Suitable for interactive jobs (e.g., Spark Shell, Notebooks).
  - Easier debugging with local driver logs.
- **Example Command (YARN Client)**:

```bash
spark-submit --master yarn --deploy-mode client my_app.py
```

- **Drawback**: Job fails if the driver machine is slow or loses connection.

### Deployment Mode Comparison Table

| **Mode** | **Driver Location** | **Best For**             | **Example**                           |
|----------|---------------------|--------------------------|---------------------------------------|
| **Local** | Local machine       | Testing, debugging       | `--master local[*]`                   |
| **Client** | Local machine       | Interactive, quick runs  | `--deploy-mode client`                |
| **Cluster** | Cluster worker node | Production, batch jobs   | `--deploy-mode cluster`               |




