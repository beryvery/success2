"""구글 드라이브 업로드 (서비스 계정).

인증 정보는 환경변수에서만 읽는다 (디스크에 키를 쓰지 않음):
  GDRIVE_SERVICE_ACCOUNT_JSON  : 서비스 계정 키 JSON 문자열 전체
  GDRIVE_SERVICE_ACCOUNT_FILE  : 또는 키 JSON 파일 경로
업로드할 폴더는 서비스 계정 이메일(client_email)에 '편집자' 로 공유되어 있어야 한다.
주의: 개인 계정(My Drive)에서는 서비스 계정이 새 파일을 만들 수 없다(저장 용량 없음). 같은 이름의 파일이 이미 있으면
그 파일의 내용을 교체하는 방식(소유자는 그대로)으로 올리므로, 처음 한 번은 폴더에 같은 이름의 빈 파일을 만들어 두거나
공유 드라이브를 쓴다.
사용:
  python -m editpipe.drive <폴더ID> 파일1 [파일2 ...]
"""
from __future__ import annotations

import json
import mimetypes
import os
import sys
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive"]


def _credentials():
    from google.oauth2 import service_account

    raw = os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON")
    path = os.environ.get("GDRIVE_SERVICE_ACCOUNT_FILE")
    if raw:
        info = json.loads(raw)
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    if path and Path(path).exists():
        return service_account.Credentials.from_service_account_file(path, scopes=SCOPES)
    raise RuntimeError("드라이브 인증 정보 없음: 환경변수 GDRIVE_SERVICE_ACCOUNT_JSON 또는 GDRIVE_SERVICE_ACCOUNT_FILE 설정 필요")


def available() -> bool:
    return bool(os.environ.get("GDRIVE_SERVICE_ACCOUNT_JSON") or os.environ.get("GDRIVE_SERVICE_ACCOUNT_FILE"))


def service():
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=_credentials(), cache_discovery=False)


def list_folder(folder_id: str, svc=None) -> list[dict]:
    svc = svc or service()
    res = svc.files().list(q=f"'{folder_id}' in parents and trashed = false", fields="files(id,name,size,mimeType,webViewLink)",
                           supportsAllDrives=True, includeItemsFromAllDrives=True, pageSize=200).execute()
    return res.get("files", [])


def upload(path: Path, folder_id: str, svc=None, name: str | None = None, replace: bool = True) -> dict:
    """파일 하나 업로드. replace=True 면 같은 이름의 기존 파일은 새 버전으로 덮어씀. → {id, name, webViewLink}"""
    from googleapiclient.http import MediaFileUpload

    svc = svc or service()
    path = Path(path)
    name = name or path.name
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    media = MediaFileUpload(str(path), mimetype=mime, resumable=path.stat().st_size > 5 * 1024 * 1024)
    fields = "id,name,size,webViewLink"
    existing = [f for f in list_folder(folder_id, svc) if f["name"] == name] if replace else []
    if existing:
        return svc.files().update(fileId=existing[0]["id"], media_body=media, fields=fields, supportsAllDrives=True).execute()
    meta = {"name": name, "parents": [folder_id]}
    try:
        return svc.files().create(body=meta, media_body=media, fields=fields, supportsAllDrives=True).execute()
    except Exception as e:   # googleapiclient.errors.HttpError
        if "storageQuotaExceeded" in str(e) or "storage quota" in str(e):
            raise RuntimeError(
                f"서비스 계정은 개인(My Drive) 폴더에 새 파일을 만들 수 없습니다 (구글 정책: 저장 용량 없음). "
                f"해결: ① 드라이브에서 같은 이름('{name}')의 빈 파일을 폴더에 먼저 만들어 두면 내용만 교체해 올립니다 "
                f"② 또는 공유 드라이브(Workspace) 폴더를 사용하세요.") from e
        raise


def upload_many(paths: list[Path], folder_id: str) -> list[dict]:
    svc = service()
    return [upload(p, folder_id, svc) for p in paths if Path(p).exists()]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        print(__doc__)
        return 2
    folder, files = argv[0], [Path(p) for p in argv[1:]]
    for r in upload_many(files, folder):
        print(f"업로드 완료: {r['name']} ({int(r.get('size', 0)) / 1e6:.1f}MB) → {r.get('webViewLink')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
