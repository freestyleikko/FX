"""取引時間帯の制御。

週末ノーポジションルール:
日本時間の金曜 17:00 から月曜 7:00 までは、週明けの窓開け(ギャップ)リスクを
避けるためポジションを一切持たない。

- `in_weekend_flat_window(dt)`  : その時刻が「ノーポジション時間帯」か
- `should_liquidate_for_weekend(dt, ...)` : 決済期限が近いか(実運用のループ用)

日足バックテストでは Backtester がこのルールを
「金曜の引けで全決済・金曜は新規建てなし」として近似する。
"""

from __future__ import annotations

from datetime import datetime, timedelta

FRIDAY = 4
SATURDAY = 5
SUNDAY = 6
MONDAY = 0


def in_weekend_flat_window(
    dt_jst: datetime,
    friday_close_hour: int = 17,
    monday_open_hour: int = 7,
) -> bool:
    """日本時間 dt_jst が「ポジションを持ってはいけない時間帯」なら True。

    時間帯: 金曜 friday_close_hour:00 〜 月曜 monday_open_hour:00(JST)
    """
    wd = dt_jst.weekday()
    if wd == FRIDAY:
        return dt_jst.hour >= friday_close_hour
    if wd in (SATURDAY, SUNDAY):
        return True
    if wd == MONDAY:
        return dt_jst.hour < monday_open_hour
    return False


def should_liquidate_for_weekend(
    dt_jst: datetime,
    lead_minutes: int = 30,
    friday_close_hour: int = 17,
    monday_open_hour: int = 7,
) -> bool:
    """決済期限(金曜17:00 JST)まで lead_minutes 以内なら True。

    実運用ループで「そろそろ全決済すべきか」を判定するためのヘルパー。
    既にノーポジション時間帯に入っている場合も True を返す。
    """
    if in_weekend_flat_window(dt_jst, friday_close_hour, monday_open_hour):
        return True
    deadline = dt_jst + timedelta(minutes=lead_minutes)
    return in_weekend_flat_window(deadline, friday_close_hour, monday_open_hour)
