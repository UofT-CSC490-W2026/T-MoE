from typing import Dict, Type, TypeVar, Callable
import warnings

T = TypeVar("T")


class Registry:
    """Generic registry for components."""

    _registries: Dict[str, Dict[str, Type]] = {}

    def __init__(self, name: str):
        self.name = name
        if name not in Registry._registries:
            Registry._registries[name] = {}

    @property
    def _registry(self) -> Dict[str, Type]:
        return Registry._registries[self.name]

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a class."""

        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                warnings.warn(
                    f"Overwriting existing registration for '{name}' in {self.name} registry"
                )
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type[T]:
        """Get a registered class by name."""
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. Available: {available}"
            )
        return self._registry[name]

    def list(self) -> list:
        """List all registered names."""
        return list(self._registry.keys())

    def items(self):
        """Iterate over (name, class) pairs."""
        return self._registry.items()

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', items={self.list()})"


# Component registries
RouterRegistry = Registry("routers")
ModelRegistry = Registry("models")
ExpertRegistry = Registry("experts")
