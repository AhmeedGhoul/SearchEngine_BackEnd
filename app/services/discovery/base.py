from abc import ABC, abstractmethod
from typing import Any


class DiscoveryProvider(ABC):
    @abstractmethod
    async def search(self, keywords: list[str] | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        pass
