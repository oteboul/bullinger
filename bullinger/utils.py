from typing import Iterable
import datetime
import collections
import portion
import numpy as np
import pandas as pd


def parse_duration(duration: str) -> float:
    if isinstance(duration, float):
        return duration

    ref = '00:00:00.000'
    time_format = '%H:%M:%S.%f'
    d1 = datetime.datetime.strptime(duration, time_format)
    d2 = datetime.datetime.strptime(ref, time_format)
    return (d1 - d2).total_seconds()


def format_name(name: str) -> str:
    name = name.strip()
    parts = []
    for part in name.split():
        if part[-1].upper() == part[-1]:
            parts.append(part[:-1].title() + part[-1])
        else:
            parts.append(part.title())
    return ' '.join(parts)