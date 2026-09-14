#!/usr/bin/env python3
"""
Jarvis Car — Backend Python (Google Cloud Run / Railway)
Services: chat (OpenCode Go LLM + fallback data-driven), presence (Bluetooth), trips, weather, passengers.
Stockage: Supabase (OAuth Google clain.pierre@gmail.com).
LLM: OpenCode Go (sk-Ii1...DXb2) avec x-opencode-session header.
"""

import os, json, logging, hashlib, time, uuid, base64
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from urllib.request import urlopen, Request
from urllib.error import URLError

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client, Client
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jarvis-car")

app = FastAPI(title="Jarvis Car API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Configuration ─────────────────────────────────────────────────────────────
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
OPENCODE_GO_KEY = os.getenv("OPENCODE_GO_KEY", "sk-Ii1uc7DMZWqI3OWlXk6rptDFBgrKfmAa3hy6mqCPsUqV62r5JHtGUqUrmTuwDXb2")
OPENCODE_GO_BASE_URL = os.getenv("OPENCODE_GO_BASE_URL", "https://opencode.ai/zen/go/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FALLBACK_DATA_DRIVEN = os.getenv("FALLBACK_DATA_DRIVEN", "true").lower() == "true"

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase connecté")
    except Exception as e:
        logger.warning(f"Supabase non connecté: {e}")

# ─── Modèles Pydantic ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: Optional[Dict[str, Any]] = None
    passenger_id: Optional[str] = None
    session_id: Optional[str] = None
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    intent: str = "default"
    used_llm: bool = True
    model: str = ""
    provider: str = ""
    ok: bool = True

class PresenceHeartbeat(BaseModel):
    device_id: str
    device_name: Optional[str] = None
    passenger_id: Optional[str] = None
    rssi: Optional[int] = None
    timestamp: Optional[float] = None
    wifi_ssid: Optional[str] = None
    wifi_bssid: Optional[str] = None

class PresenceResponse(BaseModel):
    passengers: List[Dict[str, Any]]
    status: str = "ok"

class TripStart(BaseModel):
    start_location: Optional[str] = None
    passenger_ids: Optional[List[str]] = None

class TripEnd(BaseModel):
    end_location: Optional[str] = None
    distance_km: Optional[float] = None

class TripResponse(BaseModel):
    trip_id: str
    status: str

class StatsResponse(BaseModel):
    total_trips: int = 0
    avg_duration_min: float = 0
    avg_distance_km: float = 0
    common_passengers: List[str] = []
    trips_last_30_days: int = 0
    hourly_distribution: Dict[str, int] = {}

class WeatherResponse(BaseModel):
    location: str = ""
    temperature_c: float = 0
    description: str = ""
    wind_kmh: float = 0
    humidity_pct: int = 0

class StreamChunk(BaseModel):
    content: str = ""
    done: bool = False

# ─── OpenCode Go LLM Client ────────────────────────────────────────────────────
class OpenCodeGoClient:
    """Client pour l'API OpenCode Go (OpenCode Zen)."""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat_completion(self, model: str, messages: List[Dict], max_tokens: int = 1024,
                        stream: bool = False, temperature: float = 0.7) -> Optional[Dict]:
        """Appelle l'API OpenCode Go chat/completions."""
        url = f"{self.base_url}/chat/completions"
        session_id = f"jarvis-car-{uuid.uuid4().hex[:12]}"

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "stream_options": {"include_usage": True} if stream else None,
        }
        # Supprimer les champs None
        payload = {k: v for k, v in payload.items() if v is not None}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "x-opencode-session": session_id,  # Requis par OpenCode Go
            "User-Agent": "jarvis-car/1.0 (Android app)",
            "Content-Type": "application/json",
        }

        try:
            req = Request(url, data=json.dumps(payload).encode("utf-8"),
                          headers=headers, method="POST")
            if stream:
                return self._stream_response(req)
            else:
                with urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                    if "choices" in data and data["choices"]:
                        return {
                            "content": data["choices"][0]["message"]["content"],
                            "model": data.get("model", model),
                            "usage": data.get("usage", {}),
                            "id": data.get("id", ""),
                            "created": data.get("created", 0),
                        }
                    elif "error" in data:
                        logger.warning(f"OpenCode Go error: {data['error']}")
                        return None
        except Exception as e:
            logger.warning(f"OpenCode Go call failed: {e}")
            return None
        return None

    def _stream_response(self, request: Request) -> Optional[Dict]:
        """Gère les réponses streamées (SSE-like)."""
        try:
            with urlopen(request, timeout=30) as resp:
                chunks = []
                for line in resp:
                    line = line.decode("utf-8", errors="replace").strip()
                    if line.startswith("data: "):
                        chunk_data = line[6:]
                        if chunk_data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_data)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if delta.get("content"):
                                    chunks.append(delta["content"])
                        except json.JSONDecodeError:
                            pass
                return {"content": "".join(chunks), "streamed": True}
        except Exception as e:
            logger.warning(f"OpenCode Go stream failed: {e}")
            return None

    def simple_chat(self, prompt: str, model: str = "deepseek-v4-flash") -> Optional[str]:
        """Appel simple non-streaming pour le backend."""
        result = self.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stream=False,
        )
        if result and result.get("content"):
            return result["content"]
        return None


# Instancier le client OpenCode Go
opencode_go_client = OpenCodeGoClient(OPENCODE_GO_KEY, OPENCODE_GO_BASE_URL)
logger.info(f"OpenCode Go client initialisé (base_url={OPENCODE_GO_BASE_URL})")


# ─── Fonctions utilitaires ──────────────────────────────────────────────────────
def _call_opencode_go(prompt: str, model: str = "deepseek-v4-flash",
                      stream: bool = False) -> Optional[str]:
    """Wrapper pour appeler OpenCode Go."""
    if stream:
        return opencode_go_client.chat_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            stream=True,
        )
    else:
        return opencode_go_client.simple_chat(prompt, model)


def _call_gemini(prompt: str, api_key: str) -> Optional[str]:
    """Appelle Gemini (fallback)."""
    if not api_key:
        return None
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.7},
        }
        req = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            if "candidates" in data and data["candidates"]:
                cand = data["candidates"][0]
                if "content" in cand and "parts" in cand["content"]:
                    return "".join(p.get("text", "") for p in cand["content"]["parts"])
    except Exception as e:
        logger.warning(f"Gemini call failed: {e}")
    return None


def get_llm_response(message: str, context: Optional[Dict] = None,
                     passenger_id: Optional[str] = None,
                     session_id: Optional[str] = None,
                     stream: bool = False) -> ChatResponse:
    """
    Obtient une réponse IA.
    1. OpenCode Go (clé configurée)
    2. Gemini (fallback)
    3. Fallback data-driven
    """
    prompt = message
    if context:
        ctx_str = ", ".join(f"{k}={v}" for k, v in context.items() if v)
        if ctx_str:
            prompt = f"Contexte: {ctx_str}\n\nQuestion: {message}"

    # 1. OpenCode Go
    if OPENCODE_GO_KEY and OPENCODE_GO_KEY.startswith("sk-"):
        model = "deepseek-v4-flash"  # Modèle par défaut OpenCode Go
        response = _call_opencode_go(prompt, model=model, stream=stream)
        if response:
            return ChatResponse(
                response=response,
                intent=detect_intent(message, response),
                used_llm=True,
                model=model,
                provider="opencode-go",
                ok=True,
            )

    # 2. Gemini fallback
    if GEMINI_API_KEY:
        response = _call_gemini(prompt, GEMINI_API_KEY)
        if response:
            return ChatResponse(
                response=response,
                intent=detect_intent(message, response),
                used_llm=True,
                model="gemini-1.5-flash",
                provider="gemini",
                ok=True,
            )

    # 3. Fallback data-driven
    if FALLBACK_DATA_DRIVEN:
        data_response = data_driven_fallback(message, context)
        return ChatResponse(
            response=data_response,
            intent=detect_intent(message, data_response),
            used_llm=False,
            model="-",
            provider="data-driven",
            ok=True,
        )

    return ChatResponse(
        response="Le service est temporairement indisponible.",
        intent="error",
        used_llm=False,
        model="-",
        provider="none",
        ok=False,
    )


def detect_intent(message: str, response: str) -> str:
    """Détecte l'intention de la question."""
    m = message.lower()
    if any(w in m for w in ["bonjour", "salut", "hello", "hey"]):
        return "greeting"
    if any(w in m for w in ["passager", "qui est", "qui est là", "présent", "bluetooth", "appareil"]):
        return "passenger"
    if any(w in m for w in ["trajet", "voyage", "déplacement", "stats", "statistique", "nombre de trajet"]):
        return "trip_stats"
    if any(w in m for w in ["météo", "temps", "temperature", "pluie", "soleil"]):
        return "weather"
    if any(w in m for w in ["conseil", "suggestion", "aide", "help"]):
        return "advice"
    return "default"


def data_driven_fallback(message: str, context: Optional[Dict]) -> str:
    """Réponse sans LLM, basée sur contexte et règles."""
    m = message.lower()

    if "bonjour" in m or "salut" in m:
        return "Bonjour ! Je suis Jarvis, votre assistant voiture. Je peux vous aider sur les passagers, les trajets, la météo ou toute autre question."
    if "passager" in m or "qui est" in m:
        return "Je détecte les passagers par Bluetooth. Dites-moi qui est dans la voiture ou demandez-moi 'liste des passagers'."
    if "trajet" in m or "voyage" in m:
        return "Je peux vous donner les statistiques de vos trajets. Dites-moi 'stats trajets' pour les voir."
    if "météo" in m or "temps" in m:
        return "Je peux vous donner la météo actuelle. Dites-moi 'météo' ou 'quelle température'."
    if "conseil" in m or "suggestion" in m:
        return "Voici quelques conseils: maintenez une distance de sécurité, adaptez votre vitesse au trafic, et prenez des pauses toutes les 2 heures. Besoin d'autre chose ?"
    if "help" in m or "aide" in m:
        return "Je suis Jarvis. Vous pouvez me demander: liste des passagers, stats trajets, météo, conseils de conduite, ou toute autre question."
    return f"Je suis Jarvis. Je n'ai pas compris votre question: '{message}'. Essayez: 'liste des passagers', 'stats trajets', 'météo', ou 'conseil'."


# ─── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "jarvis-car", "opencode_go": bool(OPENCODE_GO_KEY and OPENCODE_GO_KEY.startswith("sk-"))}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, x_api_key: Optional[str] = Header(None)):
    """Conversation IA — OpenCode Go LLM avec fallback data-driven."""
    result = get_llm_response(req.message, req.context, req.passenger_id, req.session_id)
    # Persister le message si Supabase
    if supabase:
        try:
            supabase.table("chat_history").insert({
                "message": req.message,
                "response": result.response,
                "passenger_id": req.passenger_id,
                "session_id": req.session_id or str(uuid.uuid4()),
                "used_llm": result.used_llm,
                "intent": result.intent,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception as e:
            logger.warning(f"Persist chat failed: {e}")
    return result


@app.post("/chat/stream")
def chat_stream(req: ChatRequest, x_api_key: Optional[str] = Header(None)):
    """Conversation IA avec streaming (SSE)."""
    prompt = req.message
    if req.context:
        ctx_str = ", ".join(f"{k}={v}" for k, v in req.context.items() if v)
        if ctx_str:
            prompt = f"Contexte: {ctx_str}\n\nQuestion: {req.message}"

    if OPENCODE_GO_KEY and OPENCODE_GO_KEY.startswith("sk-"):
        model = "deepseek-v4-flash"
        result = _call_opencode_go(prompt, model=model, stream=True)
        if result and result.get("content"):
            def event_generator():
                yield json.dumps({"content": result["content"], "done": True}) + "\n"
            return StreamingResponse(event_generator(), media_type="application/x-ndjson")
        else:
            # Fallback data-driven si le stream échoue
            fallback = data_driven_fallback(req.message, req.context)
            def fallback_generator():
                yield json.dumps({"content": fallback, "done": True}) + "\n"
            return StreamingResponse(fallback_generator(), media_type="application/x-ndjson")

    # Fallback data-driven si OpenCode Go non configuré
    fallback = data_driven_fallback(req.message, req.context)
    def fallback_generator():
        yield json.dumps({"content": fallback, "done": True}) + "\n"
    return StreamingResponse(fallback_generator(), media_type="application/x-ndjson")


@app.post("/presence", response_model=PresenceResponse)
def presence_heartbeat(req: PresenceHeartbeat):
    """Heartbeat Bluetooth/Wi-Fi d'un appareil passager."""
    now = time.time()
    ts = req.timestamp or now
    device_id = req.device_id
    device_name = req.device_name or f"Appareil-{device_id[:8]}"
    passenger_id = req.passenger_id

    # Lookup passager connu
    passenger_info = {}
    if passenger_id and supabase:
        try:
            res = supabase.table("passengers").select("*").eq("id", passenger_id).execute()
            if res.data:
                passenger_info = res.data[0]
        except Exception as e:
            logger.warning(f"Passenger lookup failed: {e}")

    # Enregistrer présence
    presence_record = {
        "device_id": device_id,
        "device_name": device_name,
        "rssi": req.rssi,
        "wifi_ssid": req.wifi_ssid,
        "wifi_bssid": req.wifi_bssid,
        "passenger_id": passenger_id,
        "passenger_name": passenger_info.get("name", "") if passenger_info else "",
        "is_known_passenger": bool(passenger_id),
        "timestamp": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
    }
    if supabase:
        try:
            supabase.table("presence").insert(presence_record).execute()
        except Exception as e:
            logger.warning(f"Persist presence failed: {e}")

    passengers_present = []
    if passenger_id:
        passengers_present.append({
            "device_id": device_id,
            "device_name": device_name,
            "passenger_id": passenger_id,
            "passenger_name": passenger_info.get("name", "Inconnu"),
            "is_known": True,
            "last_seen": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        })

    return PresenceResponse(passengers=passengers_present, status="ok")


@app.get("/presence", response_model=PresenceResponse)
def get_presence(current_time: Optional[float] = None):
    """Liste des passagers présents (dans les 5 dernières minutes)."""
    cutoff = (time.time() - 300) if current_time is None else (current_time - 300)
    passengers = []
    if supabase:
        try:
            res = supabase.table("presence")\
                .select("device_id, device_name, passenger_id, passenger_name, is_known_passenger, timestamp")\
                .gte("timestamp", datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat())\
                .order("timestamp", desc=True)\
                .execute()
            seen = {}
            for row in res.data:
                did = row["device_id"]
                if did not in seen:
                    seen[did] = row
                    passengers.append({
                        "device_id": row["device_id"],
                        "device_name": row["device_name"],
                        "passenger_id": row.get("passenger_id"),
                        "passenger_name": row.get("passenger_name", ""),
                        "is_known": row.get("is_known_passenger", False),
                        "last_seen": row["timestamp"],
                    })
        except Exception as e:
            logger.warning(f"Get presence failed: {e}")
    return PresenceResponse(passengers=passengers, status="ok")


@app.post("/trips/start", response_model=TripResponse)
def trip_start(req: TripStart, x_api_key: Optional[str] = Header(None)):
    """Début d'un trajet."""
    trip_id = str(uuid.uuid4())
    record = {
        "trip_id": trip_id,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "start_location": req.start_location,
        "passenger_ids": req.passenger_ids or [],
        "end_time": None,
        "end_location": None,
        "distance_km": None,
        "duration_min": None,
        "status": "en_cours",
    }
    if supabase:
        try:
            supabase.table("trips").insert(record).execute()
        except Exception as e:
            logger.warning(f"Persist trip start failed: {e}")
    return TripResponse(trip_id=trip_id, status="en_cours")


@app.post("/trips/end", response_model=TripResponse)
def trip_end(req: TripEnd, trip_id: Optional[str] = Header(None), x_api_key: Optional[str] = Header(None)):
    """Fin d'un trajet."""
    if not trip_id:
        raise HTTPException(400, "trip_id header required")
    now = datetime.now(timezone.utc)
    updates = {
        "end_time": now.isoformat(),
        "end_location": req.end_location,
        "distance_km": req.distance_km,
    }
    if supabase:
        try:
            supabase.table("trips").update(updates).eq("trip_id", trip_id).execute()
        except Exception as e:
            logger.warning(f"Persist trip end failed: {e}")
    return TripResponse(trip_id=trip_id, status="terminé")


@app.get("/stats", response_model=StatsResponse)
def get_stats(x_api_key: Optional[str] = Header(None)):
    """Statistiques de trajets (dernières 30 jours)."""
    stats = StatsResponse()
    if supabase:
        try:
            res = supabase.table("trips")\
                .select("*")\
                .gte("start_time", (datetime.now(timezone.utc) - timedelta(days=30)).isoformat())\
                .execute()
            trips = res.data if res.data else []
            stats.total_trips = len(trips)

            durations = []
            distances = []
            for t in trips:
                if t.get("start_time") and t.get("end_time"):
                    try:
                        s = datetime.fromisoformat(t["start_time"])
                        e = datetime.fromisoformat(t["end_time"])
                        durations.append((e - s).total_seconds() / 60)
                    except Exception:
                        pass
                if t.get("distance_km"):
                    distances.append(t["distance_km"])
            if durations:
                stats.avg_duration_min = round(sum(durations) / len(durations), 1)
            if distances:
                stats.avg_distance_km = round(sum(distances) / len(distances), 2)

            passager_counts: Dict[str, int] = {}
            for t in trips:
                for pid in t.get("passenger_ids", []):
                    passager_counts[pid] = passager_counts.get(pid, 0) + 1
            stats.common_passengers = sorted(passager_counts, key=passager_counts.get, reverse=True)[:5]

            hourly: Dict[str, int] = {}
            for t in trips:
                try:
                    s = datetime.fromisoformat(t["start_time"])
                    h = s.strftime("%H:00")
                    hourly[h] = hourly.get(h, 0) + 1
                except Exception:
                    pass
            stats.hourly_distribution = hourly

        except Exception as e:
            logger.warning(f"Stats query failed: {e}")
    return stats


@app.get("/weather", response_model=WeatherResponse)
def get_weather(latitude: Optional[float] = None, longitude: Optional[float] = None,
                location: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    """Météo via Open-Meteo (gratuit, pas de clé)."""
    lat, lon = latitude, longitude
    if lat is None or lon is None:
        lat, lon = 48.8566, 2.3522  # Paris par défaut
        if location:
            try:
                url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json&limit=1"
                req = Request(url, headers={"User-Agent": "JarvisCar/1.0"})
                with urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read())
                    if data:
                        lat = float(data[0]["lat"])
                        lon = float(data[0]["lon"])
            except Exception:
                pass

    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=auto"
        req = Request(url, headers={"User-Agent": "JarvisCar/1.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            cw = data.get("current_weather", {})
            temp = cw.get("temperature", 0)
            wind = cw.get("windspeed", 0)
            weather_code = cw.get("weathercode", 0)
            desc = _weather_code_to_desc(weather_code)
            return WeatherResponse(
                location=f"{lat:.4f}, {lon:.4f}",
                temperature_c=temp,
                description=desc,
                wind_kmh=wind,
                humidity_pct=0,
            )
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return WeatherResponse(location="inconnue", temperature_c=0, description="Erreur météo")


def _weather_code_to_desc(code: int) -> str:
    codes = {
        0: "Ciel dégagé",
        1: "Peu nuageux",
        2: "Nuageux",
        3: "Couvert",
        45: "Brumeux",
        48: "Brume",
        51: "Brouillard léger",
        53: "Brouillard",
        55: "Brouillard dense",
        61: "Pluie légère",
        63: "Pluie modérée",
        65: "Pluie forte",
        71: "Neige légère",
        73: "Neige modérée",
        75: "Neige forte",
        80: "Averses légères",
        81: "Averses de pluie",
        82: "Averses de pluie fortes",
        95: "Orage",
    }
    return codes.get(code, "Conditions inconnues")


@app.get("/passengers", response_model=List[Dict])
def list_passengers(x_api_key: Optional[str] = Header(None)):
    if supabase:
        try:
            res = supabase.table("passengers").select("*").order("name").execute()
            return res.data or []
        except Exception as e:
            logger.warning(f"List passengers failed: {e}")
    return []


@app.post("/passengers", response_model=Dict)
def create_passenger(passenger: Dict, x_api_key: Optional[str] = Header(None)):
    if not supabase:
        raise HTTPException(503, "Supabase not configured")
    try:
        res = supabase.table("passengers").insert(passenger).execute()
        return res.data[0] if res.data else {}
    except Exception as e:
        raise HTTPException(500, str(e))


# ─── Démarrage ──────────────────────────────────────────────────────────────────
def get_app():
    return app

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
