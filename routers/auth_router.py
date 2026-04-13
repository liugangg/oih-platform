"""
auth_router — Optional multi-user authentication for OIH.

Features:
  - Admin-provisioned accounts (no self-registration)
  - Per-user chat history isolation
  - Task queries scoped by user_id
  - CLI management: python auth_router.py add|list|reset|disable|enable <args>

Set OIH_JWT_SECRET env var before production use.
"""

import os, sys, sqlite3
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ── Config ─────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("OIH_JWT_SECRET", "CHANGE_ME_set_OIH_JWT_SECRET_env_var")
ALGORITHM  = "HS256"
TOKEN_DAYS = 30
DB_PATH    = "/data/oih/oih-api/data/oih_users.db"

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer  = HTTPBearer(auto_error=False)


# ── DB bootstrap ───────────────────────────────────────────────────────
def get_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            is_active     INTEGER DEFAULT 1,
            created_at    TEXT    DEFAULT (datetime('now')),
            last_login    TEXT
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id         TEXT    PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            title      TEXT,
            svc        TEXT,
            created_at TEXT    DEFAULT (datetime('now')),
            updated_at TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
        CREATE TABLE IF NOT EXISTS chat_messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            session_id TEXT    NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            tool_name  TEXT,
            created_at TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_chat_user_session
            ON chat_messages(user_id, session_id);
    """)
    conn.commit()
    return conn


# ── JWT helpers ────────────────────────────────────────────────────────
def create_token(user_id: int, username: str) -> str:
    exp = datetime.utcnow() + timedelta(days=TOKEN_DAYS)
    return jwt.encode(
        {"sub": str(user_id), "username": username, "exp": exp},
        SECRET_KEY, algorithm=ALGORITHM
    )

def decode_token(token: str) -> dict:
    try:
        p = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"user_id": int(p["sub"]), "username": p["username"]}
    except JWTError:
        raise HTTPException(401, "Token 无效或已过期，请重新登录")

def get_current_user(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not creds:
        raise HTTPException(401, "请先登录")
    return decode_token(creds.credentials)


# ── Middleware ─────────────────────────────────────────────────────────
class UserContextMiddleware(BaseHTTPMiddleware):
    """Attach user_id/username to request.state for task filtering."""
    async def dispatch(self, request: Request, call_next):
        request.state.user_id = None
        request.state.username = None
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            try:
                info = decode_token(auth[7:])
                request.state.user_id  = info["user_id"]
                request.state.username = info["username"]
            except HTTPException:
                pass
        return await call_next(request)


# ── Schemas ────────────────────────────────────────────────────────────
class LoginReq(BaseModel):
    username: str
    password: str

class AuthResp(BaseModel):
    token: str
    username: str
    user_id: int

class SaveMsgReq(BaseModel):
    session_id: str
    role: str           # 'user' | 'assistant' | 'tool_call' | 'tool_result'
    content: str
    tool_name: Optional[str] = None
    svc: Optional[str] = None   # 'small' | 'adc' | 'gromacs'


# ── Auth endpoints ─────────────────────────────────────────────────────
@router.post("/login", response_model=AuthResp)
def login(req: LoginReq):
    db = get_db()
    try:
        row = db.execute(
            "SELECT * FROM users WHERE username=? AND is_active=1",
            (req.username.strip(),)
        ).fetchone()
        if not row or not pwd_ctx.verify(req.password, row["password_hash"]):
            raise HTTPException(401, "用户名或密码错误")
        db.execute(
            "UPDATE users SET last_login=datetime('now') WHERE id=?", (row["id"],)
        )
        db.commit()
        return AuthResp(
            token=create_token(row["id"], row["username"]),
            username=row["username"],
            user_id=row["id"]
        )
    finally:
        db.close()

@router.get("/me")
def me(user=Depends(get_current_user)):
    return user


# ── Session & history endpoints ────────────────────────────────────────
@router.get("/sessions")
def list_sessions(user=Depends(get_current_user)):
    db = get_db()
    try:
        rows = db.execute(
            "SELECT id, title, svc, created_at, updated_at FROM sessions "
            "WHERE user_id=? ORDER BY updated_at DESC LIMIT 100",
            (user["user_id"],)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()

@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, user=Depends(get_current_user)):
    db = get_db()
    try:
        sess = db.execute(
            "SELECT id FROM sessions WHERE id=? AND user_id=?",
            (session_id, user["user_id"])
        ).fetchone()
        if not sess:
            raise HTTPException(403, "无权访问该会话")
        rows = db.execute(
            "SELECT role, content, tool_name, created_at FROM chat_messages "
            "WHERE session_id=? AND user_id=? ORDER BY id ASC",
            (session_id, user["user_id"])
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()

@router.post("/sessions/message")
def save_message(req: SaveMsgReq, user=Depends(get_current_user)):
    db = get_db()
    try:
        existing = db.execute(
            "SELECT id FROM sessions WHERE id=? AND user_id=?",
            (req.session_id, user["user_id"])
        ).fetchone()
        if not existing:
            title = req.content[:40] + ("…" if len(req.content) > 40 else "")
            db.execute(
                "INSERT INTO sessions (id, user_id, title, svc) VALUES (?,?,?,?)",
                (req.session_id, user["user_id"], title, req.svc)
            )
        else:
            db.execute(
                "UPDATE sessions SET updated_at=datetime('now') WHERE id=?",
                (req.session_id,)
            )
        db.execute(
            "INSERT INTO chat_messages (user_id, session_id, role, content, tool_name) "
            "VALUES (?,?,?,?,?)",
            (user["user_id"], req.session_id, req.role, req.content, req.tool_name)
        )
        db.commit()
        return {"ok": True}
    finally:
        db.close()

@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, user=Depends(get_current_user)):
    db = get_db()
    try:
        db.execute(
            "DELETE FROM chat_messages WHERE session_id=? AND user_id=?",
            (session_id, user["user_id"])
        )
        db.execute(
            "DELETE FROM sessions WHERE id=? AND user_id=?",
            (session_id, user["user_id"])
        )
        db.commit()
        return {"ok": True}
    finally:
        db.close()


# ── Task isolation helper ──────────────────────────────────────────────
def filter_tasks_by_user(tasks: list, user_id: int) -> list:
    """
    Use in existing task list endpoints to scope results per user.
    Tasks with no user_id field are legacy/admin — skip or show to all.
    Example in tasks router:
        from routers.auth_router import filter_tasks_by_user, UserContextMiddleware
        tasks = filter_tasks_by_user(all_tasks, request.state.user_id)
    """
    return [t for t in tasks
            if t.get("user_id") is None or t.get("user_id") == user_id]


# ── Admin CLI ──────────────────────────────────────────────────────────
def cli():
    import argparse
    parser = argparse.ArgumentParser(description="OIH User Management")
    sub = parser.add_subparsers(dest="cmd")

    a = sub.add_parser("add");     a.add_argument("username"); a.add_argument("password")
    sub.add_parser("list")
    r = sub.add_parser("reset");   r.add_argument("username"); r.add_argument("new_password")
    d = sub.add_parser("disable"); d.add_argument("username")
    e = sub.add_parser("enable");  e.add_argument("username")

    args = parser.parse_args()
    db   = get_db()

    if args.cmd == "add":
        try:
            db.execute(
                "INSERT INTO users (username, password_hash) VALUES (?,?)",
                (args.username, pwd_ctx.hash(args.password))
            )
            db.commit()
            print(f"✅ 用户 '{args.username}' 已创建")
        except sqlite3.IntegrityError:
            print(f"❌ 用户名 '{args.username}' 已存在")

    elif args.cmd == "list":
        rows = db.execute(
            "SELECT id, username, is_active, created_at, last_login FROM users ORDER BY id"
        ).fetchall()
        print(f"{'ID':<4} {'用户名':<20} {'状态':<8} {'创建时间':<22} 最后登录")
        print("─" * 72)
        for r in rows:
            st = "✅启用" if r["is_active"] else "🚫禁用"
            print(f"{r['id']:<4} {r['username']:<20} {st:<8} "
                  f"{(r['created_at'] or '-'):<22} {r['last_login'] or '-'}")

    elif args.cmd == "reset":
        c = db.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (pwd_ctx.hash(args.new_password), args.username)
        ).rowcount
        db.commit()
        print(f"✅ 密码已重置" if c else f"❌ 用户不存在")

    elif args.cmd == "disable":
        db.execute("UPDATE users SET is_active=0 WHERE username=?", (args.username,))
        db.commit(); print(f"🚫 '{args.username}' 已禁用")

    elif args.cmd == "enable":
        db.execute("UPDATE users SET is_active=1 WHERE username=?", (args.username,))
        db.commit(); print(f"✅ '{args.username}' 已启用")

    else:
        parser.print_help()

    db.close()


if __name__ == "__main__":
    cli()
