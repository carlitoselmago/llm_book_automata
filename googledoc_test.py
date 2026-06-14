from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
from time import sleep

creds = Credentials.from_service_account_file(
    'service_account.json',
    scopes=['https://www.googleapis.com/auth/documents']
)

service = build('docs', 'v1', credentials=creds)
# from the URL
doc_id = '1SKnwnakyP2xdRKnJI6xh3P9qSaWxWQI23EF_lVNxDhU'  

requests = [
    {
        'insertText': {
            'location': {'index': 1}, 
            'text': 'No volveré a copiar en clase!\n'
        }
    }
]
for i in range(100):
    service.documents().batchUpdate(
        documentId=doc_id,
        body={'requests': requests}
    ).execute()
    sleep(1)