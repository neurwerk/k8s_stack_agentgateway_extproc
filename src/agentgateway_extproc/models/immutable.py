"""Read-only, serializable maps for trusted policy snapshots."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated

from pydantic import AfterValidator, SerializerFunctionWrapHandler, WrapSerializer


def _freeze[K, V](value: Mapping[K, V]) -> Mapping[K, V]:
    return MappingProxyType(dict(value))


def _serialize[K, V](value: Mapping[K, V], handler: SerializerFunctionWrapHandler) -> object:
    return handler(dict(value))


type FrozenMap[K, V] = Annotated[
    Mapping[K, V],
    AfterValidator(_freeze),
    WrapSerializer(_serialize),
]
