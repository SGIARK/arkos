def get_from_google_calendar(calendar_id):
    url = f"https://www.googleapis.com/calendar/v3/users/me/calendarList/{calendar_id}"
    headers = {"Content-Type": "application/json"}
    data = {"contextHistory": [], "dc": ""}
    response = requests.post(url, headers=headers, json=data)
    res_data = json.loads(response.text)  # returns all possible queries
    pass


if __name__ is "__main__":

    pass
