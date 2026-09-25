"""EventBus asynchrone (README §5) : découple producteurs et consommateurs.

    @bus.subscribe(EventType.TRADE)
    async def on_trade(event): ...

    await bus.publish(event)

Les handlers d'un même événement sont appelés séquentiellement, dans l'ordre
d'abonnement : le traitement est déterministe, donc reproductible. Une
exception dans un handler est journalisée et n'empêche pas les autres
handlers de recevoir l'événement.
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable, Union

from trading_engine.data.events import Event, EventType

logger = logging.getLogger(__name__)

Handler = Callable[[Event], Union[Awaitable[Any], Any]]

ALL = "*"


class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self.published_count = 0
        self.error_count = 0

    @staticmethod
    def _key(event_type: EventType | str) -> str:
        return event_type.value if isinstance(event_type, EventType) else event_type

    def subscribe(self, event_type: EventType | str = ALL, handler: Handler | None = None):
        """Abonne `handler` ; utilisable directement ou comme décorateur.

        `event_type="*"` reçoit tous les événements (ex. stockage).
        """
        key = self._key(event_type)

        def register(fn: Handler) -> Handler:
            self._handlers[key].append(fn)
            return fn

        return register(handler) if handler is not None else register

    def unsubscribe(self, event_type: EventType | str, handler: Handler) -> None:
        self._handlers[self._key(event_type)].remove(handler)

    async def publish(self, event: Event) -> None:
        self.published_count += 1
        handlers = self._handlers.get(event.event_type.value, []) + self._handlers.get(ALL, [])
        for handler in handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                self.error_count += 1
                logger.exception(
                    "handler %s failed on %s event", getattr(handler, "__name__", handler),
                    event.event_type.value,
                )
