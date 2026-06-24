import os
import json
import flask
from flask import Flask, redirect, url_for, session, render_template_string
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.auth.transport.requests

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-me")

# Allow HTTP for local testing
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

CLIENT_SECRETS_FILE = "client_secret.json"

# Scopes — each one requires explicit user consent on the Google screen.
# Add or remove scopes based on what data you want to collect for the book.
SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    # YouTube: subscriptions, liked videos, watch history (via activity)
    "https://www.googleapis.com/auth/youtube.readonly",
    # People API: full profile fields
    "https://www.googleapis.com/auth/user.birthday.read",
    "https://www.googleapis.com/auth/user.addresses.read",
    "https://www.googleapis.com/auth/user.phonenumbers.read",
    "https://www.googleapis.com/auth/user.organization.read",
    "https://www.googleapis.com/auth/user.gender.read",
    "https://www.googleapis.com/auth/contacts.readonly",  # full address book
]

HOME_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Data Book Project</title>
  <style>
    body { }
    h1 { font-size: 1.8em; margin-bottom: 0.3em; }
    p { color: #555; margin-bottom: 2em; }
    .btn {
      display: inline-flex; align-items: center; gap: 12px;
      background: #fff; border: 1px solid #ddd; border-radius: 4px;
      padding: 12px 24px; font-size: 1em; cursor: pointer;
      text-decoration: none; color: #333; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .btn:hover { box-shadow: 0 2px 6px rgba(0,0,0,0.15); }
    .btn img { width: 20px; height: 20px; }
  </style>
</head>
<body>
 
  <a href="/login" class="btn">
    <img src="https://developers.google.com/identity/images/g-logo.png" alt="G">
    Sign in with Google
  </a>
</body>
</html>
"""

PROFILE_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Data collected</title>
  <style>
    body { font-family: monospace; max-width: 800px; margin: 40px auto; background: #111; color: #0f0; padding: 20px; }
    h2 { color: #fff; }
    pre { white-space: pre-wrap; word-break: break-word; }
    a { color: #0af; }
  </style>
</head>
<body>
  <h2>Data collected — thank you</h2>
  <pre>{{ data }}</pre>
  <br><a href="/logout">Sign out</a>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HOME_HTML)


@app.route("/login")
def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri=url_for("callback", _external=True),
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        code_challenge_method="S256",
    )
    session["state"] = state
    session["code_verifier"] = flow.code_verifier
    return redirect(auth_url)


@app.route("/callback")
def callback():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        state=session["state"],
        redirect_uri=url_for("callback", _external=True),
    )
    flow.fetch_token(
        authorization_response=flask.request.url,
        code_verifier=session["code_verifier"],
    )
    credentials = flow.credentials

    collected = {}

    # Basic profile
    user_info_service = build("oauth2", "v2", credentials=credentials)
    user_info = user_info_service.userinfo().get().execute()
    collected["profile"] = user_info

    # YouTube data
    try:
        yt = build("youtube", "v3", credentials=credentials)

        subs = yt.subscriptions().list(
            part="snippet", mine=True, maxResults=50
        ).execute()
        collected["youtube_subscriptions"] = [
            item["snippet"]["title"] for item in subs.get("items", [])
        ]

        liked = yt.videos().list(
            part="snippet",
            myRating="like",
            maxResults=50,
        ).execute()
        collected["youtube_liked_videos"] = [
            {"title": v["snippet"]["title"], "channel": v["snippet"]["channelTitle"]}
            for v in liked.get("items", [])
        ]

        activity = yt.activities().list(
            part="snippet,contentDetails", mine=True, maxResults=50
        ).execute()
        collected["youtube_activity"] = [
            {
                "type": a["snippet"]["type"],
                "title": a["snippet"].get("title", ""),
                "date": a["snippet"].get("publishedAt", ""),
            }
            for a in activity.get("items", [])
        ]
    except Exception as e:
        collected["youtube_error"] = str(e)

    # People API — maximum available fields
    try:
        people = build("people", "v1", credentials=credentials)

        # Own profile with every available personField
        ALL_PERSON_FIELDS = ",".join([
            "addresses", "ageRanges", "biographies", "birthdays",
            "calendarUrls", "clientData", "coverPhotos", "emailAddresses",
            "events", "externalIds", "genders", "imClients", "interests",
            "locales", "locations", "memberships", "metadata",
            "miscKeywords", "names", "nicknames", "occupations",
            "organizations", "phoneNumbers", "photos", "relations",
            "sipAddresses", "skills", "urls", "userDefined",
        ])

        profile = people.people().get(
            resourceName="people/me",
            personFields=ALL_PERSON_FIELDS,
        ).execute()
        collected["people_profile"] = profile

        # Full address book (contacts) — paginate to get all
        contacts = []
        page_token = None
        while True:
            kwargs = dict(
                resourceName="people/me",
                personFields=ALL_PERSON_FIELDS,
                pageSize=1000,
            )
            if page_token:
                kwargs["pageToken"] = page_token
            resp = people.people().connections().list(**kwargs).execute()
            contacts.extend(resp.get("connections", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        collected["people_contacts"] = contacts
        collected["people_contacts_total"] = len(contacts)

        # Other contacts (suggested / non-connection contacts)
        try:
            other = people.otherContacts().list(
                readMask="names,emailAddresses,phoneNumbers,photos",
                pageSize=1000,
            ).execute()
            collected["people_other_contacts"] = other.get("otherContacts", [])
        except Exception as e:
            collected["people_other_contacts_error"] = str(e)

    except Exception as e:
        collected["people_error"] = str(e)

    # Persist to a JSON file (one file per user)
    user_id = user_info.get("id", "unknown")
    os.makedirs("collected_data", exist_ok=True)
    out_path = os.path.join("collected_data", f"{user_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(collected, f, indent=2, ensure_ascii=False)

    return render_template_string(PROFILE_HTML, data=json.dumps(collected, indent=2, ensure_ascii=False))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="localhost")
