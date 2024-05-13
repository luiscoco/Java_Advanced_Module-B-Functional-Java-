# Java Advanced Module-B: Executor framework Fork Join pool

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

This example demonstrates scheduling a task with a delay

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

This example demonstrates how to use a Callable task that returns a result. The Future object can be used to retrieve this result

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

 I can provide more examples to help you understand different features of the Java Executor Framework. Let's explore some practical use cases and advanced features of the framework, including handling multiple futures, combining results, and handling exceptions.

**Example 4: Handling Multiple Futures**

This example demonstrates how to manage a list of Future objects, allowing you to execute multiple tasks and process their results as they become available

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

This example shows how to combine results from several Callable tasks that return values, such as calculating the sum of returned integers

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

This example illustrates how to handle exceptions that occur within Callable tasks

It demonstrates proper error handling without terminating the program abruptly

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

Here's an example using RecursiveAction for performing a simple task, like incrementing all elements of an array

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

Here's an example using RecursiveTask to compute a value, such as finding the sum of an array

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

These examples demonstrate the basic usage of the Fork/Join Framework, which is very effective for tasks that can be broken down recursively

By understanding how to create and manage tasks with RecursiveAction and RecursiveTask, developers can effectively utilize multicore processors to enhance performance for suitable tasks
