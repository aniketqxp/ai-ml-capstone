# Repository Architecture

## System Components

* **backend/**: Core FastAPI application container. Manages database transactions, coordinates downstream model inferences, and handles analytics dashboard data payloads.
* **frontend/**: Client-side React and Tailwind CSS single-page web application. Provides dashboard views, processing state polling, and synchronized audio playback interfaces.
* **ml-services/**: Isolated data science laboratory and modeling execution workspace. Contains development environments, model evaluation utilities, and production script paths.

---

## Directory Tree Layout

```text
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml             # Automated code quality, syntax checking, and linting gates
│
├── backend/                      # API Infrastructure & Downstream Routing
│   ├── app/
│   │   ├── api/                  # REST route handlers (batch job triggers, dashboard analytics)
│   │   ├── database.py           # PostgreSQL engine initialization and session management
│   │   ├── main.py               # Application entry point, CORS configurations, and routers
│   │   ├── models.py             # Relational SQL table definitions (calls, transcripts, evidence)
│   │   │
│   │   ├── agents/               # Multi-Agent Sequential Logic Domain
│   │   │   ├── pipeline.py       # Central orchestrator executing the model and agent chain
│   │   │   ├── workflow_agent.py # Rule-based deterministic compliance matching execution module
│   │   │   ├── llm_client.py     # Ollama controller formatting context blocks for Llama 3.2 3B
│   │   │   └── extraction.py     # JSON schema enforcement validator for structured evidence
│   │   │
│   │   └── services/             # Core Backend Utility Drivers
│   │       ├── alignment.py      # Combines transcription words with speaker turn boundaries
│   │       └── batch_service.py  # System file utility reading local AppTek records from disk
│   │
│   ├── requirements.txt          # API dependencies (fastapi, sqlalchemy, pydantic, psycopg2)
│   └── Dockerfile                # Production multi-stage build configuration for backend runtime
│
├── frontend/                     # User Interface Management
│   ├── src/
│   │   ├── components/           # Reusable UI modules (compliance trackers, audio stream layers)
│   │   ├── pages/                # Top-level application interfaces (Dashboard views, Reports panel)
│   │   ├── App.jsx               # Client application routing configuration
│   │   └── main.jsx              # DOM mounting and application bootstrap entry point
│   ├── package.json              # Client runtime scripts and node dependency versions
│   └── Dockerfile                # Node container configuration for server-side frontend builds
│
├── ml-services/                  # Machine Learning R&D and Production Execution
│   ├── notebooks/                # Research & Prototyping Sandbox
│   │   ├── 01_data_prep.ipynb    # Slicing, cleaning, and preprocessing AppTek and CREMA-D targets
│   │   ├── 02_diarization.ipynb  # Hyperparameter exploration for pyannote speaker separation
│   │   ├── 03_wav2vec2.ipynb     # Acoustic emotion classification model training and tracking
│   │   └── 04_whisper.ipynb      # Precision benchmarking for optimized 8-bit integer transcription
│   │
│   ├── src/                      # Frozen Production Modules (Refactored from Notebooks)
│   │   ├── __init__.py
│   │   ├── diarizer.py           # Instantiated class wrapping production pyannote clusters
│   │   ├── classifier.py         # Instantiated class wrapping fine-tuned Wav2Vec2 inference
│   │   └── transcriber.py        # Instantiated class wrapping optimized faster-whisper inference
│   │
│   ├── data/                     # Local data cache path for unindexed source datasets (Git-ignored)
│   ├── models/                   # Local binary model weights and checkpoint cache (Git-ignored)
│   ├── requirements.txt          # Data science libraries (torch, transformers, faster-whisper, wandb)
│   └── Dockerfile                # Image blueprint provisioning isolated Jupyter servers and ML dependencies
│
├── docker-compose.yml            # Multi-container multi-zone development orchestration matrix
└── README.md                     # System technical architecture configuration blueprint
```
