import typing


class Calendar:
    """
    class for creating calendar objects
    """

    def __init__(
        self,
        all_events: list,
    ):
        """
        init method for calendars
        """
        names_as_keys, event_as_keys = self.make_dicts(all_events)
        self.names_as_keys = names_as_keys
        self.event_as_keys = event_as_keys

    def make_dicts(self, events):
        """
        creates dictionaries

        """
        name_as_keys_dict = {}
        event_as_keys_dict = {}
        for event in events:  # iterates through all events provided

            curr_name: str = event.get_name()
            if curr_name not in name_as_keys_dict.keys():
                name_as_keys_dict[curr_name] = list()
            name_as_keys_dict[curr_name] = name_as_keys_dict[curr_name].append(
                event
            )  # appends event to dictionary list value according to name key

            if event not in event_as_keys_dict.keys():
                event_as_keys_dict[event] = list()
            event_as_keys_dict[event] = event_as_keys_dict[event].append(
                event
            )  # appends event to dictionary list value according to event key

        return [name_as_keys_dict, event_as_keys_dict]


class Event:
    """
    class for creating event objects

    """

    def __init__(self, name: str, start: str, end: str, dynammic=False):
        """
        init method for Events
        """
        self.start = start
        self.end = end
        self.name = name
        self.complete = False
        self.dynamic = dynammic

    def get_name(self):
        """
        returns name of Event
        """
        return self.name

    def get_time(self):
        """
        returns start and end time of event in the format
        ((YYYY-MM-DD H:M:S),(YYYY-MM-DD H:M:S)))
        """

        return (self.start, self.end)
