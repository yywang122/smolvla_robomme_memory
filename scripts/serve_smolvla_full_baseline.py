import argparse
import asyncio
import http
import logging
import sys
import time
import traceback

import numpy as np
import torch
import websockets.asyncio.server as ws_server
import websockets.frames

# ---------------------------------------------------------
# Reuse the original RoboMME / OpenPI msgpack implementation
# without modifying robomme_policy_learning.
# ---------------------------------------------------------
OPENPI_CLIENT_SRC = (
    "/home/cluster/robomme_policy_learning/packages/openpi-client/src"
)
if OPENPI_CLIENT_SRC not in sys.path:
    sys.path.insert(0, OPENPI_CLIENT_SRC)

from openpi_client import msgpack_numpy

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors


DEFAULT_CKPT = (
    "/home/cluster/smolvla_robomme_memory/"
    "outputs/smolvla_baseline_30k/"
    "checkpoints/030000/pretrained_model"
)


class SmolVLARoboMMEServer:
    def __init__(
        self,
        checkpoint: str,
        host: str = "0.0.0.0",
        port: int = 8011,
        exec_horizon: int = 16,
        device: str = "cuda",
    ):
        self.host = host
        self.port = port
        self.exec_horizon = exec_horizon
        self.device = device

        print("=" * 70)
        print("Loading SmolVLA")
        print("Checkpoint:", checkpoint)
        print("Device    :", device)
        print("=" * 70)

        self.policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self.policy.to(device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=checkpoint,
        )

        self.policy.reset()

        print("SmolVLA loaded.")
        print("chunk_size    :", self.policy.config.chunk_size)
        print("n_action_steps:", self.policy.config.n_action_steps)
        print("n_obs_steps   :", self.policy.config.n_obs_steps)
        print("exec_horizon  :", self.exec_horizon)
        print("=" * 70)

    def reset(self):
        self.policy.reset()

    @torch.no_grad()
    def infer(self, obs):
        """
        RoboMME input:
            observation/image       : H x W x 3 uint8
            observation/wrist_image : H x W x 3 uint8
            observation/state       : (8,)
            prompt                  : str

        Output:
            actions : (exec_horizon, 8)
        """

        front = np.asarray(obs["observation/image"])
        wrist = np.asarray(obs["observation/wrist_image"])
        state = np.asarray(
            obs["observation/state"],
            dtype=np.float32,
        )
        prompt = str(obs["prompt"])

        # SmolVLA / LeRobot expects CHW float image tensors.
        front_t = (
            torch.from_numpy(front.copy())
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        wrist_t = (
            torch.from_numpy(wrist.copy())
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        state_t = torch.from_numpy(state.copy()).float()

        batch = {
            "observation.images.image": front_t,
            "observation.images.wrist_image": wrist_t,
            "observation.state": state_t,
            "task": prompt,
        }

        # Processor adds batch dimension, language tokens, normalization, etc.
        batch = self.preprocessor(batch)

        t0 = time.monotonic()

        action_chunk = self.policy.predict_action_chunk(batch)

        infer_ms = (time.monotonic() - t0) * 1000.0

        # Convert model output back to environment action space.
        action_chunk = self.postprocessor(action_chunk)

        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.detach().cpu().numpy()
        else:
            action_chunk = np.asarray(action_chunk)

        # Expected:
        #   [1, chunk, 8]
        if action_chunk.ndim == 3:
            action_chunk = action_chunk[0]

        if action_chunk.ndim != 2:
            raise RuntimeError(
                f"Unexpected action shape: {action_chunk.shape}"
            )

        actions = action_chunk[: self.exec_horizon].astype(
            np.float32,
            copy=False,
        )

        print(
            f"[infer] prompt={prompt!r} "
            f"input={front.shape} "
            f"chunk={action_chunk.shape} "
            f"send={actions.shape} "
            f"time={infer_ms:.1f} ms"
        )

        return {
            "actions": actions,
            "inference_time_ms": np.float32(infer_ms),
        }


async def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(
            http.HTTPStatus.OK,
            "OK\n",
        )
    return None


async def handler(websocket, server):
    print(
        "[connect]",
        websocket.remote_address,
    )

    packer = msgpack_numpy.Packer()

    # Original RoboMME client expects metadata immediately after connecting.
    metadata = {
        "policy": "SmolVLA",
        "baseline": "no_memory",
        "exec_horizon": server.exec_horizon,
    }

    await websocket.send(
        packer.pack(metadata)
    )

    while True:
        try:
            raw = await websocket.recv()
            obs = msgpack_numpy.unpackb(raw)

            if obs.get("reset", False):
                t0 = time.monotonic()

                server.reset()

                reset_ms = (
                    time.monotonic() - t0
                ) * 1000.0

                print("[reset] policy queue cleared")

                await websocket.send(
                    packer.pack(
                        {
                            "reset_finished": True,
                            "reset_time_ms": reset_ms,
                        }
                    )
                )
                continue

            # Baseline intentionally does NOT support/use memory buffer.
            if obs.get("add_buffer", False):
                await websocket.send(
                    packer.pack(
                        {
                            "add_buffer_finished": True,
                            "ignored": True,
                        }
                    )
                )
                continue

            output = server.infer(obs)

            await websocket.send(
                packer.pack(output)
            )

        except websockets.ConnectionClosed:
            print(
                "[disconnect]",
                websocket.remote_address,
            )
            break

        except Exception:
            tb = traceback.format_exc()
            print(tb)

            await websocket.send(tb)

            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="SmolVLA inference error",
            )
            raise


async def main_async(args):
    server = SmolVLARoboMMEServer(
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        exec_horizon=args.exec_horizon,
        device=args.device,
    )

    print()
    print(
        f"SmolVLA RoboMME server listening on "
        f"ws://{args.host}:{args.port}"
    )
    print("Waiting for RoboMME client...")
    print()

    async with ws_server.serve(
        lambda ws: handler(ws, server),
        args.host,
        args.port,
        compression=None,
        max_size=None,
        process_request=health_check,
    ) as ws:
        await ws.serve_forever()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CKPT,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8011,
    )
    parser.add_argument(
        "--exec-horizon",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
