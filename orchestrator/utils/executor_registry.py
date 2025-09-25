class ExecutorRegistry:
    """Registry for managing thread pool executor classes.

    This class provides a centralized registry for tracking which classes
    require thread pool executors. It allows classes to register themselves and
    provides validation functionality to check if a class is registered. This
    is useful for managing shared thread pool resources across different
    components in the orchestrator system.
    """

    executor_registry = set()

    @staticmethod
    def register_class(class_name: str) -> None:
        """Register a class name in the executor registry.

        Adds a class name to the registry to indicate that this class
        requires a thread pool executor. This is typically called during
        class definition or initialization.

        Args:
            class_name (str):
                The name of the class to register.
        """
        ExecutorRegistry.executor_registry.add(class_name)

    @staticmethod
    def validate_class(class_name: str) -> bool:
        """Validate if a class is registered in the executor registry.

        Checks whether a given class name has been registered in the
        executor registry. This is used to determine if a class requires
        a thread pool executor.

        Args:
            class_name (str):
                The name of the class to validate.

        Returns:
            bool:
                True if the class is registered, False otherwise.
        """
        return class_name in ExecutorRegistry.executor_registry
