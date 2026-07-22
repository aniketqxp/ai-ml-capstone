# Resolución de conflictos: PR #160 (Rodrigo) vs PR #159 (Asma)

## Archivos resueltos en esta carpeta

- `backend/app/models.py` → se tomó la versión de Asma completa (agrega
  `SentimentSegment`, `CallAudioSummary`, relaja `call_id` a opcional en
  `Transcript`/`Evaluation` y agrega `source_call_id`).
- `backend/app/routers/calls.py` → se tomó la versión de Asma completa
  (`/summary`, `/transcripts/ingest`, `/evaluate`, `/{call_id}/flags`).
  Tu versión solo tenía comentarios placeholder, sin lógica real que
  conservar.
- `backend/app/routers/agents.py`, `backend/app/routers/sentiment.py`,
  `backend/app/main.py` → no existían en tu rama o no tenían conflicto,
  se copian tal cual de Asma.
- `docker-compose.yml` → fusionado: se mantiene el enfoque de Asma
  (`.env` para `DATABASE_URL`), pero el servicio `db` local se conserva
  como perfil opcional (`docker compose --profile local-db up`) por si
  alguien quiere desarrollar sin depender del Postgres compartido.

**Cómo aplicar:** copia estos archivos sobre las rutas equivalentes en tu
rama `deployment` antes de resolver los conflictos que marca GitHub en
el PR #160, o pégalos directamente al resolver el conflicto en la UI de
GitHub.

## Lo que NO pude resolver por falta de información

Tu `CompliancePanel` (dentro de `App.jsx`) espera esta forma de datos:

```js
evaluation.compliance.name_announced.passed / .evidence.sec
evaluation.quality.efficiency  // 1-5
evaluation.escalation.risk_level  // 'none' | 'review' | 'escalate'
evaluation.escalation.customer_emotion_text
```

El modelo de Asma solo define columnas planas: `overall_grade`,
`escalation_risk` (entero 0-10), `scorecard` (JSONB) y
`compliance_flags` (JSONB) — sin fijar qué claves va adentro de esos dos
JSONB. No tengo un ejemplo real de esos payloads, así que no puedo
armar el adaptador sin adivinar.

**Antes de tu próximo commit, pregúntale a Asma (o revisa el pipeline de
Aniket, que es quien llena `/calls/evaluate`):**
1. ¿Qué claves exactas mete en `scorecard` (JSONB)? ¿Ahí van los 6
   puntos de compliance y las 6 dimensiones de calidad?
2. ¿`escalation_risk` (0-10) se traduce a los tres niveles que usa tu
   UI (`none`/`review`/`escalate`) con algún corte fijo, o hay que
   calcularlo tú mismo en el frontend?

Con esas dos respuestas puedo escribirte la función `mapEvaluation()`
que traduce la fila real de Supabase al formato que ya consume tu
`CompliancePanel`, sin tocar el resto del componente.

## Nota aparte (no bloquea el merge, pero vale la pena limpiar)

`AudioPlayer.jsx`, `ComplianceScorecard.jsx` y `TranscriptionView.jsx`
no están importados en ningún lado — toda la funcionalidad real vive
inline en `App.jsx`. La descripción del PR #160 dice que los "agrega",
pero en la práctica quedaron huérfanos. Antes de pedir review, decide
si los conectas de verdad o los borras y ajustas la descripción del PR.
