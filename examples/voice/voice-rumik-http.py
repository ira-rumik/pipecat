#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
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
from pipecat.services.rumik import RumikHttpTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


def optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def optional_float_env(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
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


MUGA_SYSTEM_INSTRUCTION = (
    "You are speaking through Rumik AI's muga TTS. Reply in expressive, natural "
    "Roman Hinglish, like a warm voice agent talking live. Start every response "
    "with exactly one tone tag: [happy], [excited], [sad], [angry], [neutral], "
    "or [whisper], followed by one space. Pick a tone that matches the user's "
    "emotion, and use <laugh>, <chuckle>, or <sigh> sparingly only when it fits "
    "the tone. Keep replies to one or two short sentences, ideally 10-30 words, "
    "so they are easy to speak. No emojis, markdown, bullets, Devanagari, or "
    "unsupported tags."
)


def create_stt() -> DeepgramSTTService:
    return DeepgramSTTService(
        api_key=os.environ["DEEPGRAM_API_KEY"],
        settings=DeepgramSTTService.Settings(
            model="nova-3-general",
            language="multi",
        ),
    )


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info(f"Starting bot")

    async with aiohttp.ClientSession() as session:
        stt = create_stt()

        tts = RumikHttpTTSService(
            api_key=os.environ["RUMIK_API_KEY"],
            gateway_url=os.environ["RUMIK_GATEWAY_URL"],
            aiohttp_session=session,
            settings=RumikHttpTTSService.Settings(
                model=os.getenv("RUMIK_MODEL", "muga"),
                voice=os.getenv("RUMIK_SPEAKER") or None,
                description=os.getenv("RUMIK_DESCRIPTION") or None,
                f0_up_key=optional_int_env("RUMIK_F0_UP_KEY"),
                temperature=optional_float_env("RUMIK_TEMPERATURE"),
                top_p=optional_float_env("RUMIK_TOP_P"),
                top_k=optional_int_env("RUMIK_TOP_K"),
                repetition_penalty=optional_float_env("RUMIK_REPETITION_PENALTY"),
                max_new_tokens=optional_int_env("RUMIK_MAX_NEW_TOKENS"),
            ),
        )

        llm = OpenAILLMService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAILLMService.Settings(
                system_instruction=MUGA_SYSTEM_INSTRUCTION,
            ),
        )

        context = LLMContext()
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
        )

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                user_aggregator,  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                assistant_aggregator,  # Assistant spoken responses
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
            logger.info(f"Client connected")
            # Kick off the conversation.
            context.add_message(
                {"role": "developer", "content": "Please introduce yourself to the user."}
            )
            await worker.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
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
