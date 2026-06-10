#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.rumik import RumikTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


def optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def create_stt() -> DeepgramSTTService:
    return DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            language="multi",
        ),
    )


MULBERRY_VOICE_DESCRIPTION = (
    "expressive warm Indian conversational voice, friendly and emotionally present, "
    "with natural pauses, gentle emphasis, and a polished voice-agent delivery"
)

MULBERRY_SYSTEM_INSTRUCTION = (
    "You are speaking through Rumik AI's Mulberry TTS. Reply in natural Roman "
    "Hinglish with expressive, human phrasing. Keep responses to one or two short "
    "sentences, ideally 10-30 words, so they sound good when spoken. Match the "
    "user's emotion with warmth, curiosity, and subtle emphasis in the wording. "
    "Do not use Muga tone tags like [happy], event tags like <laugh>, emojis, "
    "markdown, bullets, or Devanagari."
)


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting Rumik AI mulberry voice bot")

    stt = create_stt()

    tts = RumikTTSService(
        api_key=os.environ["RUMIK_API_KEY"],
        gateway_url=os.environ["RUMIK_GATEWAY_URL"],
        settings=RumikTTSService.Settings(
            model="mulberry",
            voice=os.getenv("RUMIK_SPEAKER") or "speaker_1",
            description=os.getenv("RUMIK_DESCRIPTION") or MULBERRY_VOICE_DESCRIPTION,
            f0_up_key=optional_int_env("RUMIK_F0_UP_KEY") or 10,
        ),
    )

    llm = OpenAILLMService(
        api_key=os.environ["OPENAI_API_KEY"],
        settings=OpenAILLMService.Settings(
            system_instruction=MULBERRY_SYSTEM_INSTRUCTION,
        ),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        context.add_message(
            {"role": "developer", "content": "Please introduce yourself to the user."}
        )
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
