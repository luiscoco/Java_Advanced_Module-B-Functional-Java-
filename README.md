# Java Advanced Module-B: Executor framework and Fork Join pool

## 1. Executor Framework in Java

The **Executor Framework** in **Java** is part of the **java.util.concurrent** package, introduced in **Java 5**

It helps in managing threads efficiently through a set of high-level APIs for running tasks asynchronously

The framework simplifies task execution by abstracting thread creation, life cycle management, and other complexities of threading

Let's dive into some of the key components and see simple examples for each:

**Key Components**

**Executor**: This is the basic interface that supports launching new tasks

**ExecutorService**: A subinterface of Executor, it provides lifecycle methods to manage termination and methods producing Futures for tracking progress of tasks

**ScheduledExecutorService**: An interface that can schedule tasks to run after a delay or periodically

**Thread Pool**: A group of pre-instantiated reusable threads. Classes like ThreadPoolExecutor and Executors provide factory methods to create standardized thread pools

**Simple Examples**

**Example 1: Using ExecutorService**

This example uses an ExecutorService to run a simple task asynchronously.

```java
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SimpleExecutorServiceExample {
    public static void main(String[] args) {
        // Creating an ExecutorService with a fixed thread pool of 2 threads.
        ExecutorService executor = Executors.newFixedThreadPool(2);

        // Runnable task that prints the current thread name
        Runnable task = () -> {
            System.out.println("Current thread: " + Thread.currentThread().getName());
        };

        // Submit the task to the executor
        executor.execute(task);

        // Shutdown the executor
        executor.shutdown();
    }
}
```

**Example 2: Using ScheduledExecutorService**

This example demonstrates **scheduling a task with a delay**

```java
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.android.util.concurrent.TimeUnit;

public class ScheduledExecutorExample {
    public static void main(String[] args) {
        // Creating a ScheduledExecutorService
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        // Runnable task that prints the current date
        Runnable task = () -> System.out.println("Executing at: " + new Date());

        // Schedule the task with a 5-second delay
        scheduler.schedule(task, 5, TimeUnit.SECONDS);

        // Shutdown the executor after some time
        scheduler.schedule(() -> scheduler.shutdown(), 10, TimeUnit.SECONDS);
    }
}
```

**Example 3: Future and Callable**

This example demonstrates how to **use a Callable task that returns a result**

The **Future** object can be used to retrieve this result

```java
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class CallableAndFutureExample {
    public static void main(String[] args) throws InterruptedException, ExecutionException {
        // Create an ExecutorService
        ExecutorService executor = Executors.newCachedThreadPool();

        // Callable task that returns the current thread name
        Callable<String> task = () -> {
            return "Executed by: " + Thread.currentThread().getName();
        };

        // Submit the callable task
        Future<String> future = executor.submit(task);

        // Get the result of the callable
        String result = future.get();  // This line will block until the result is available
        System.out.println(result);

        // Shutdown the executor
        executor.shutdown();
    }
}
```

These examples illustrate basic usage of the Executor Framework in Java

They demonstrate how to set up an executor, run tasks asynchronously, schedule tasks, and retrieve results from tasks that compute values

This framework is highly useful for handling complex multithreading scenarios with relative ease

I can provide more examples to help you understand different features of the Java Executor Framework

Let's explore some practical use cases and advanced features of the framework, including handling multiple futures, combining results, and handling exceptions

**Example 4: Handling Multiple Futures**

This example demonstrates how to **manage a list of Future** objects, allowing you to **execute multiple tasks** and **process their results** as they become available

```java
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MultipleFuturesExample {
    public static void main(String[] args) throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(4);
        List<Future<String>> futures = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            int taskId = i;
            Callable<String> task = () -> "Task " + taskId + " executed";
            futures.add(executor.submit(task));
        }

        for (Future<String> future : futures) {
            System.out.println(future.get());  // Waits for the task to complete and prints the result
        }

        executor.shutdown();
    }
}
```

**Example 5: Combine Results from Callables**

This example shows how to **combine results from several Callable tasks** that return values, such as calculating the **sum of returned integers**

```java
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class CombineResultsExample {
    public static void main(String[] args) throws Exception {
        ExecutorService executor = Executors.newCachedThreadPool();
        List<Future<Integer>> futures = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
            int value = i * 10;
            Callable<Integer> task = () -> {
                Thread.sleep(100); // Simulate some computation
                return value + 10;
            };
            futures.add(executor.submit(task));
        }

        int sum = 0;
        for (Future<Integer> future : futures) {
            sum += future.get();
        }

        System.out.println("Total sum: " + sum);

        executor.shutdown();
    }
}
```

**Example 6: Handling Exceptions in Callable Tasks**

This example illustrates how to **handle exceptions** that occur within Callable tasks

It demonstrates proper **error handling** without terminating the program abruptly

```java
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ExceptionHandlingExample {
    public static void main(String[] args) {
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Callable<Integer> task = () -> {
            throw new IllegalStateException("I throw an exception!");
        };

        Future<Integer> future = executor.submit(task);

        try {
            future.get();
        } catch (ExecutionException ee) {
            System.err.println("Task threw an exception!");
            System.err.println(ee.getCause());
        } catch (InterruptedException ie) {
            Thread.currentThread().interrupt();  // set the interrupt flag
            System.err.println("Task was interrupted!");
        }

        executor.shutdown();
    }
}
```

**Example 7: Scheduled Task with Fixed Rate**

This example uses **ScheduledExecutorService** to **schedule a periodic task that executes repeatedly at fixed intervals**, regardless of the task's execution duration

```java
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class ScheduledTaskFixedRateExample {
    public static void main(String[] args) {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);

        Runnable task = () -> {
            System.out.println("Executing Task at " + System.nanoTime());
        };

        // Initialize Delay - 0, Period - 2 seconds
        scheduler.scheduleAtFixedRate(task, 0, 2, TimeUnit.SECONDS);

        // Scheduler will keep running. To stop, you might use:
        // scheduler.shutdown();
    }
}
```

These examples demonstrate how to use the Executor Framework effectively to handle multiple tasks, manage asynchronous computation, and deal with exceptions in Java, providing a robust foundation for concurrent programming in Java applications.

## 2. Fork/Join Framework in Java

The Fork/Join Framework is a part of Java's java.util.concurrent package, specifically designed to help leverage multi-core processors effectively

It is ideal for tasks that can be broken down into smaller, independent subtasks, which can be executed concurrently, and then combined to produce a final result

This is achieved using a divide-and-conquer approach

**Key Components**

**ForkJoinPool**: The heart of the Fork/Join Framework. It is an implementation of ExecutorService that manages worker threads

**RecursiveAction**: A recursive resultless ForkJoinTask (typically for side effects, such as array sorting)

**RecursiveTask**: A recursive result-bearing ForkJoinTask

**Principles of the Fork/Join Framework**

Work-Stealing Algorithm: Worker threads that run out of tasks can "steal" tasks from other threads that are still busy

Task Splitting/Forking: Large tasks are split (forked) into smaller tasks until the task size is small enough to be executed directly

Joining Tasks: Results of subtasks are joined together to form the final result

**Simple Examples**

**Example 1: Using RecursiveAction**

This code demonstrates how to use the **Fork/Join Framework** in **Java** to **increment all elements of an array** in parallel using RecursiveAction

Here’s a brief explanation of each part:

**Imports**

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
```

These imports are necessary for:

**ForkJoinPool**: The class that manages the worker threads

**RecursiveAction**: A base class for tasks that do not return a result

**Class Definition**

```java
public class SimpleRecursiveActionExample extends RecursiveAction {
    private long[] array;
    private int low;
    private int high;
    private static final int THRESHOLD = 1000; // This threshold can be adjusted based on the task size and system
```

SimpleRecursiveActionExample extends **RecursiveAction**, indicating that it **performs a task that doesn't return a result**

The class has four fields:

**array**: The array of **long integers** to be processed

**low and high**: Indices that define the **range of the array to be processed**

**THRESHOLD**: A constant that determines when to **stop splitting tasks** and start processing directly

**Constructor**

```java
public SimpleRecursiveActionExample(long[] array, int low, int high) {
    this.array = array;
    this.low = low;
    this.high = high;
}
```

This constructor initializes the fields with the provided array and the range to be processed

**Compute Method**

```java
@Override
protected void compute() {
    if (high - low < THRESHOLD) {
        for (int i = low; i < high; ++i) {
            array[i]++; // Incrementing each element by one
        }
    } else {
        int mid = (low + high) >>> 1;
        SimpleRecursiveActionExample left = new SimpleRecursiveActionExample(array, low, mid);
        SimpleRecursiveActionExample right = new SimpleRecursiveActionExample(array, mid, high);
        invokeAll(left, right);
    }
}
```

The compute method is where the task logic resides:

If the task size is below the **THRESHOLD**, it processes the elements **directly** by incrementing each element in the specified range

Otherwise, it **splits the task into two subtasks** at the midpoint and invokes them in parallel using **invokeAll**

IMPORTANT NOTE:

The line of code ```int mid = (low + high) >>> 1;``` is used to **calculate the midpoint between two indices**, low and high, in an array

Let's break down what this code does:

Adding Low and High:

```java
low + high
```

This part calculates the sum of low and high, which are the start and end indices of the portion of the array being processed.

Unsigned Right Shift (>>>):

```java
(low + high) >>> 1
```

The ```>>>``` operator is the **unsigned right shift operator**

It shifts the bits of the integer to the right by the specified number of positions, in this case, by 1 position

This effectively **divides the number by 2**.

**Main Method**

```java
public static void main(String[] args) {
    long[] array = new long[2000];
    ForkJoinPool pool = new ForkJoinPool();
    SimpleRecursiveActionExample task = new SimpleRecursiveActionExample(array, 0, array.length);
    pool.invoke(task);
}
```

Initializes an array of 2000 long integers

Creates a ForkJoinPool to manage the parallel execution

Creates and invokes a SimpleRecursiveActionExample task with the entire array range

**Summary**

This code illustrates how to use the **Fork/Join Framework** to **process an array in parallel**

The compute method **divides the task recursively** until the size of the task is manageable (less than THRESHOLD)

It then increments each element in the specified range

The main method sets up the array, creates a **ForkJoinPool**, and runs the **parallel computation**

This approach efficiently utilizes multiple cores to perform array processing tasks concurrently

**Source Code**

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class SimpleRecursiveActionExample extends RecursiveAction {
    private long[] array;
    private int low;
    private int high;
    private static final int THRESHOLD = 1000; // This threshold can be adjusted based on the task size and system

    public SimpleRecursiveActionExample(long[] array, int low, int high) {
        this.array = array;
        this.low = low;
        this.high = high;
    }

    @Override
    protected void compute() {
        if (high - low < THRESHOLD) {
            for (int i = low; i < high; ++i) {
                array[i]++; // Incrementing each element by one
            }
        } else {
            int mid = (low + high) >>> 1;
            SimpleRecursiveActionExample left = new SimpleRecursiveActive(array, low, mid);
            SimpleRecursiveActionExample right = new SimpleRecursiveActive(array, mid, high);
            invokeAll(left, right);
        }
    }

    public static void main(String[] args) {
        long[] array = new long[2000];
        ForkJoinPool pool = new ForkJoinPool();
        SimpleRecursiveActionExample task = new SimpleRecursiveActionExample(array, 0, array.length);
        pool.invoke(task);
    }
}
```

**Example 2: Using RecursiveTask**

This code demonstrates how to use the Fork/Join Framework in Java to compute the sum of elements in a large array in parallel using RecursiveTask

Here is a brief explanation of each part:

**Imports**

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
```

These imports are necessary for:

**ForkJoinPool**: The main class that manages the worker threads

**RecursiveTask**: A base class for tasks that return a result

**Class Definition**

```java
public class SimpleRecursiveTaskExample extends RecursiveTask<Long> {
    private long[] array;
    private int low;
    private int high;
    private static final int THRESHOLD = 1000;
```

**SimpleRecursiveTaskExample** extends ```RecursiveTask<Long>```, indicating that it performs a task that returns a Long result

 The class has four fields:

**array**: The array of long integers to be processed

**low and high**: Indices that define the range of the array to be processed

**THRESHOLD**: A constant that determines when to stop splitting tasks and start processing directly

**Constructor**

```java
public SimpleRecursiveTaskExample(long[] array, int low, int high) {
    this.array = array;
    this.low = low;
    this.high = high;
}
```

This constructor initializes the fields with the provided array and the range to be processed

**Compute Method**

```java
@Override
protected Long compute() {
    if (high - low < THRESHOLD) {
        long sum = 0;
        for (int i = low; i < high; ++i) {
            sum += array[i];
        }
        return sum;
    } else {
        int mid = (low + high) >>> 1;
        SimpleRecursiveTaskExample left = new SimpleRecursiveTaskExample(array, low, mid);
        SimpleRecursiveTaskExample right = new SimpleRecursiveTaskExample(array, mid, high);
        left.fork();
        long rightResult = right.compute();
        long leftResult = left.join();
        return leftResult + rightResult;
    }
}
```

The compute method contains the logic for the task:

If the size of the task (difference between high and low) is less than THRESHOLD, it **calculates the sum directly**

Otherwise, it **splits the task into two subtasks** at the midpoint and **processes them in parallel** using **fork** and **join**

**Main Method**

```java
public static void main(String[] args) {
    long[] array = new long[4000]; // large array initialization with values
    ForkJoinPool pool = new ForkJoinPool();
    SimpleRecursiveTaskExample task = new SimpleRecursiveTaskExample(array, 0, array.length);
    long sum = pool.invoke(task);
    System.out.println("Sum: " + sum);
}
```

Initializes an array of 4000 long integers

Creates a ForkJoinPool to manage the parallel execution

Creates and invokes a SimpleRecursiveTaskExample task with the entire array range

Prints the sum of the array elements

**Summary**

This code illustrates how to use the **Fork/Join Framework** to compute the **sum of an array in parallel**

The compute method **divides the task recursively** until the size of the task is manageable (less than **THRESHOLD**)

It then **calculates the sum directly** or **splits the task and processes the subtasks in parallel**

The main method sets up the array, creates a ForkJoinPool, and runs the parallel computation, efficiently utilizing multiple cores to perform the sum calculation concurrently

**Source Code** 

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class SimpleRecursiveTaskExample extends RecursiveTask<Long> {
    private long[] array;
    private int low;
    private int high;
    private static final int THRESHOLD = 1000;

    public SimpleRecursiveTaskExample(long[] array, int low, int high) {
        this.array = array;
        this.low = low;
        this.high = high;
    }

    @Override
    protected Long compute() {
        if (high - low < THRESHOLD) {
            long sum = 0;
            for (int i = low; i < high; ++i) {
                sum += array[i];
            }
            return sum;
        } else {
            int mid = (low + high) >>> 1;
            SimpleRecursiveTaskExample left = new SimpleRecursiveTaskExample(array, low, mid);
            SimpleRecursiveTaskExample right = new SimpleRecursiveTaskExample(array, mid, high);
            left.fork();
            long rightResult = right.compute();
            long leftResult = left.join();
            return leftResult + rightResult;
        }
    }

    public static void main(String[] args) {
        long[] array = new long[4000]; // large array initialization with values
        ForkJoinPool pool = new ForkJoinPool();
        SimpleRecursiveTaskExample task = new SimpleRecursiveTaskExample(array, 0, array.length);
        long sum = pool.invoke(task);
        System.out.println("Sum: " + sum);
    }
}
```

**Example 3: RecursiveTask for Parallel Array Search**

This example demonstrates how to use **RecursiveTask** to perform a parallel **search in an array to find the maximum value**

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class ParallelMaxFinder extends RecursiveTask<Integer> {
    private static final int THRESHOLD = 1000;
    private final int[] array;
    private final int start;
    private final int end;

    public ParallelMaxFinder(int[] array, int start, int end) {
        this.array = array;
        this.start = start;
        this.end = end;
    }

    @Override
    protected Integer compute() {
        if (end - start < THRESHOLD) {
            int max = array[start];
            for (int i = start + 1; i < end; i++) {
                if (array[i] > max) {
                    max = array[i];
                }
            }
            return max;
        } else {
            int mid = (start + end) / 2;
            ParallelMaxFinder leftTask = new ParallelMaxFinder(array, start, mid);
            ParallelMaxFinder rightTask = new ParallelMaxFinder(array, mid, end);
            leftTask.fork();
            int rightResult = rightTask.compute();
            int leftResult = leftTask.join();
            return Math.max(leftResult, rightResult);
        }
    }

    public static void main(String[] args) {
        int[] array = new int[10000];
        // Populate the array with random values
        for (int i = 0; i < array.length; i++) {
            array[i] = (int) (Math.random() * 10000);
        }

        ForkJoinPool pool = new ForkJoinPool();
        ParallelMaxFinder task = new ParallelMaxFinder(array, 0, array.length);
        int max = pool.invoke(task);
        System.out.println("Max value: " + max);
    }
}
```

**Example 4: RecursiveAction for Matrix Multiplication**

This example demonstrates how to use RecursiveAction for parallel matrix multiplication

The methodology for splitting the original matrix in the ParallelMatrixMultiplication code involves dividing the task into smaller subtasks recursively, where each subtask is responsible for computing a part of the resultant matrix

Let's break down the code and the splitting strategy in detail

**Original Matrix Multiplication**

The original problem is to multiply two matrices A and B and store the result in matrix C

The multiplication of two matrices involves computing the dot product of rows from matrix A with columns from matrix B

**Splitting the Matrix**

When the size of the submatrix exceeds the THRESHOLD, the task is split into four subtasks, each responsible for a quadrant of the result matrix C

The main idea is to divide the problem into smaller parts until they are small enough to be computed directly

**Splitting Methodology**

Let's consider the current submatrix defined by row, col, and size. The new size of each submatrix will be size / 2

Here is a visual representation of the splitting process for a matrix of size n x n:

Before Splitting:

```
C (size x size)
┌───────┬───────┐
│       │       │
│       │       │
│       │       │
├───────┼───────┤
│       │       │
│       │       │
│       │       │
└───────┴───────┘
```

After Splitting:

```
C (size x size)
┌─────────────┬─────────────┐
│   (row,     │   (row,     │
│    col)     │    col +    │
│             │    newSize) │
│  (newSize)  │  (newSize)  │
├─────────────┼─────────────┤
│   (row +    │   (row +    │
│    newSize, │    newSize, │
│    col)     │    col +    │
│             │    newSize) │
│  (newSize)  │  (newSize)  │
└─────────────┴─────────────┘
```

Each submatrix will have half the size of the original matrix:

**Top-left submatrix**:

```java
new ParallelMatrixMultiplication(A, B, C, row, col, newSize)
```

This subtask handles the multiplication for the top-left quadrant of C

**Top-right submatrix**:

```java
new ParallelMatrixMultiplication(A, B, C, row, col + newSize, newSize)
```

This subtask handles the multiplication for the top-right quadrant of C

**Bottom-left submatrix**:

```java
new ParallelMatrixMultiplication(A, B, C, row + newSize, col, newSize)
```

This subtask handles the multiplication for the bottom-left quadrant of C

**Bottom-right submatrix**:

```java
new ParallelMatrixMultiplication(A, B, C, row + newSize, col + newSize, newSize)
```

This subtask handles the multiplication for the bottom-right quadrant of C

**Base Case**: When the size of the submatrix is **less than or equal to THRESHOLD**, the multiplication is performed **directly** using **three nested loops**

**Recursive Case**: When the size of the submatrix is greater than THRESHOLD, the task is split into four smaller subtasks

Each **subtask** is responsible for **computing one quadrant** of the resulting matrix C

The **invokeAll** method is used to **run these subtasks in parallel**

**Recursive Task Splitting Example**

Assume the original matrix size is 512 x 512 and THRESHOLD is 64. The recursive splitting would occur as follows:

**First Split**:

Each subtask handles a 256 x 256 submatrix

**Second Split**:

Each 256 x 256 submatrix is further split into four 128 x 128 submatrices

**Third Split**:

Each 128 x 128 submatrix is split into four 64 x 64 submatrices

**No further split since 64 <= THRESHOLD**

This recursive splitting continues until each **submatrix size** is **small** enough to be **computed directly**

**Summary**

The methodology for splitting the original matrix in the ParallelMatrixMultiplication class involves dividing the task into four smaller tasks whenever the size of the current submatrix is larger than the THRESHOLD

Each subtask is responsible for computing a quadrant of the resulting matrix C

This recursive approach effectively utilizes the Fork/Join Framework to perform parallel matrix multiplication, efficiently leveraging multi-core processors

**Source code**

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class ParallelMatrixMultiplication extends RecursiveAction {
    private static final int THRESHOLD = 64;
    private final double[][] A;
    private final double[][] B;
    private final double[][] C;
    private final int row;
    private final int col;
    private final int size;

    public ParallelMatrixMultiplication(double[][] A, double[][] B, double[][] C, int row, int col, int size) {
        this.A = A;
        this.B = B;
        this.C = C;
        this.row = row;
        this.col = col;
        this.size = size;
    }

    @Override
    protected void compute() {
        if (size <= THRESHOLD) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    for (int k = 0; k < size; k++) {
                        C[row + i][col + j] += A[row + i][k] * B[k][col + j];
                    }
                }
            }
        } else {
            int newSize = size / 2;
            invokeAll(
                new ParallelMatrixMultiplication(A, B, C, row, col, newSize),
                new ParallelMatrixMultiplication(A, B, C, row, col + newSize, newSize),
                new ParallelMatrixMultiplication(A, B, C, row + newSize, col, newSize),
                new ParallelMatrixMultiplication(A, B, C, row + newSize, col + newSize, newSize)
            );
        }
    }

    public static void main(String[] args) {
        int n = 512; // Matrix size (must be a power of 2)
        double[][] A = new double[n][n];
        double[][] B = new double[n][n];
        double[][] C = new double[n][n];

        // Initialize matrices A and B with random values
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = Math.random();
                B[i][j] = Math.random();
            }
        }

        ForkJoinPool pool = new ForkJoinPool();
        ParallelMatrixMultiplication task = new ParallelMatrixMultiplication(A, B, C, 0, 0, n);
        pool.invoke(task);

        // Print the result matrix C
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%8.2f ", C[i][j]);
            }
            System.out.println();
        }
    }
}
```

**Example 5: RecursiveTask for Parallel Merge Sort**

This example demonstrates how to use RecursiveTask for implementing **parallel merge sort**

It **compares elements from the left and right arrays** and **adds the smaller element to the result array**

**After one of the arrays is fully processed**, any **remaining elements** from the other array are **added to the result array**

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;
import java.util.Arrays;

public class ParallelMergeSort extends RecursiveTask<int[]> {
    private static final int THRESHOLD = 1000;
    private final int[] array;

    public ParallelMergeSort(int[] array) {
        this.array = array;
    }

    @Override
    protected int[] compute() {
        if (array.length < THRESHOLD) {
            Arrays.sort(array);
            return array;
        } else {
            int mid = array.length / 2;
            ParallelMergeSort leftTask = new ParallelMergeSort(Arrays.copyOfRange(array, 0, mid));
            ParallelMergeSort rightTask = new ParallelMergeSort(Arrays.copyOfRange(array, mid, array.length));
            invokeAll(leftTask, rightTask);
            int[] leftResult = leftTask.join();
            int[] rightResult = rightTask.join();
            return merge(leftResult, rightResult);
        }
    }

    private int[] merge(int[] left, int[] right) {
        int[] result = new int[left.length + right.length];
        int i = 0, j = 0, k = 0;
        while (i < left.length && j < right.length) {
            if (left[i] <= right[j]) {
                result[k++] = left[i++];
            } else {
                result[k++] = right[j++];
            }
        }
        while (i < left.length) {
            result[k++] = left[i++];
        }
        while (j < right.length) {
            result[k++] = right[j++];
        }
        return result;
    }

    public static void main(String[] args) {
        int[] array = new int[10000];
        // Populate the array with random values
        for (int i = 0; i < array.length; i++) {
            array[i] = (int) (Math.random() * 10000);
        }

        ForkJoinPool pool = new ForkJoinPool();
        ParallelMergeSort task = new ParallelMergeSort(array);
        int[] sortedArray = pool.invoke(task);

        // Print the sorted array
        for (int value : sortedArray) {
            System.out.print(value + " ");
        }
    }
}
```

**Example 6: Using Work-Stealing Algorithm**

This code demonstrates the use of the Fork/Join Framework in Java to process an array of integers using the **work-stealing algorithm**

Here is a brief explanation of each part:

Imports

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;
```

These imports are necessary for:

**ForkJoinPool**: The main class that manages the **worker threads**

**RecursiveAction**: A base class for **tasks that do not return a result**

**TimeUnit**: For time-based operations, specifically used **for sleeping**

**Class Definition**

```java
public class WorkStealingDemo extends RecursiveAction {
    private static final int THRESHOLD = 10;
    private final int[] array;
    private final int start;
    private final int end;
```

**WorkStealingDemo extends RecursiveAction**, indicating that it performs a **task that doesn't return a result**

The class has three fields:

**THRESHOLD**: A constant that determines when to stop splitting tasks

**array**: The array of integers to be processed

**start and end**: Indices that define the range of the array to be processed

**Constructor**

```java
Copy code
public WorkStealingDemo(int[] array, int start, int end) {
    this.array = array;
    this.start = start;
    this.end = end;
}
```

This constructor initializes the fields with the provided array and the range to be processed

**Compute Method**

```java
@Override
protected void compute() {
    if (end - start < THRESHOLD) {
        for (int i = start; i < end; i++) {
            array[i] = process(array[i]);
        }
    } else {
        int mid = (start + end) / 2;
        WorkStealingDemo leftTask = new WorkStealingDemo(array, start, mid);
        WorkStealingDemo rightTask = new WorkStealingDemo(array, mid, end);
        invokeAll(leftTask, rightTask);
    }
}
```

The compute method is where the task logic resides:

If the task size is below the THRESHOLD, it processes the elements directly

Otherwise, it splits the task into two subtasks and invokes them in parallel using invokeAll

**Process Method**

```java
private int process(int value) {
    try {
        TimeUnit.MILLISECONDS.sleep(value);
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
    return value * 2;
}
```

The process method simulates a time-consuming task by sleeping for a number of milliseconds equal to the element's value and then doubling the value

**Main Method**

```java
public static void main(String[] args) {
    int[] array = new int[100];
    // Populate the array with values ranging from 1 to 100
    for (int i = 0; i < array.length; i++) {
        array[i] = i + 1;
    }

    ForkJoinPool pool = new ForkJoinPool();
    WorkStealingDemo task = new WorkStealingDemo(array, 0, array.length);
    pool.invoke(task);

    // Print the processed array
    for (int value : array) {
        System.out.print(value + " ");
    }
}
```

Initializes an array of 100 integers with values from 1 to 100

Creates a ForkJoinPool to manage the parallel execution

Creates and invokes a WorkStealingDemo task with the entire array range

Prints the processed array after the task completes

**Summary**

This code illustrates how to use the Fork/Join Framework to process an array in parallel, demonstrating task splitting, work-stealing, and recursive computation

The compute method splits the tasks until they are small enough to be processed directly, leveraging multiple cores for efficient processing

The process method simulates a computational task, and the main method sets up and runs the parallel computation

**Source code**

This example demonstrates the work-stealing algorithm by simulating a simple workload distribution scenario

```java
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.TimeUnit;

public class WorkStealingDemo extends RecursiveAction {
    private static final int THRESHOLD = 10;
    private final int[] array;
    private final int start;
    private final int end;

    public WorkStealingDemo(int[] array, int start, int end) {
        this.array = array;
        this.start = start;
        this.end = end;
    }

    @Override
    protected void compute() {
        if (end - start < THRESHOLD) {
            for (int i = start; i < end; i++) {
                array[i] = process(array[i]);
            }
        } else {
            int mid = (start + end) / 2;
            WorkStealingDemo leftTask = new WorkStealingDemo(array, start, mid);
            WorkStealingDemo rightTask = new WorkStealingDemo(array, mid, end);
            invokeAll(leftTask, rightTask);
        }
    }

    private int process(int value) {
        try {
            TimeUnit.MILLISECONDS.sleep(value);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return value * 2;
    }

    public static void main(String[] args) {
        int[] array = new int[100];
        // Populate the array with values ranging from 1 to 100
        for (int i = 0; i < array.length; i++) {
            array[i] = i + 1;
        }

        ForkJoinPool pool = new ForkJoinPool();
        WorkStealingDemo task = new WorkStealingDemo(array, 0, array.length);
        pool.invoke(task);

        // Print the processed array
        for (int value : array) {
            System.out.print(value + " ");
        }
    }
}
```

These examples illustrate the versatility of the Fork/Join Framework, showing how it can be used for parallel search, matrix multiplication, merge sort, and even demonstrating the work-stealing algorithm

By leveraging these capabilities, developers can efficiently utilize **multi-core processors** for complex and large-scale tasks

These examples demonstrate the basic usage of the **Fork/Join Framework**, which is very effective for **tasks that can be broken down recursively**

By understanding how to create and manage tasks with **RecursiveAction** and **RecursiveTask**, developers can effectively utilize **multicore processors** to enhance performance for suitable tasks
