"""FastAPI application factory and lifecycle management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.config import load_config
from src.inference.pipeline import InferencePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("deepfake_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: load models on startup and release on shutdown."""
    logger.info("Initializing deepfake detection backend...")
    config = load_config()
    app.state.config = config

    if getattr(app.state, "pipeline", None) is None:
        try:
            pipeline = InferencePipeline(config=config)
            app.state.pipeline = pipeline
            logger.info("Inference pipeline successfully loaded and ready on %s.", config.device)
        except Exception as e:
            logger.warning("Could not initialize full inference pipeline on startup: %s", e)
            app.state.pipeline = None
    else:
        logger.info("Using pre-injected inference pipeline.")

    yield

    logger.info("Shutting down deepfake detection backend...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    config = load_config()

    app = FastAPI(
        title="Multimodal Deepfake Detection API",
        description="Research-oriented multimodal video and audio deepfake detection backend",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Enable CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = config
    app.include_router(router)

    return app


app = create_app()
