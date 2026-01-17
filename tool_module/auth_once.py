from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar"]

flow = InstalledAppFlow.from_client_secrets_file("secrets/gcp-oauth.keys.json", SCOPES)

creds = flow.run_local_server(port=0)

with open("secrets/google_tokens.json", "w") as f:
    f.write(creds.to_json())

print("OAuth complete. Tokens saved.")
