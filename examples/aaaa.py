import debugpy

debugpy.listen(("localhost", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()

# Put your code below
print("Hello, debug!")