from random import choices, shuffle


def merge_events(events_0, events_1, ratio=0.5):
    total_len = len(events_0) + len(events_1)
    len_1 = int(total_len * ratio)
    len_0 = total_len - len_1
    events = choices(population=events_0, k=len_0) + \
        choices(population=events_1, k=len_1)
    shuffle(events)
    return events
