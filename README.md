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

## 2. Fork/Join Framework in Java


