import sys
import json
from pathlib import Path

import numpy as np

# ============================================================
# Paths
# ============================================================

# Original repos are READ ONLY.
ROBOMME_ROOT = Path("/home/cluster/robomme_policy_learning")
ROBOMME_EXAMPLES = ROBOMME_ROOT / "examples" / "robomme"
OPENPI_CLIENT = ROBOMME_ROOT / "packages" / "openpi-client" / "src"

sys.path.insert(0, str(ROBOMME_EXAMPLES))
sys.path.insert(0, str(OPENPI_CLIENT))

from env_runner import EnvRunner
from utils import RolloutRecorder
from openpi_client.websocket_client_policy import MMEVLAWebsocketClientPolicy


# ============================================================
# Config
# ============================================================

TASK = "BinFill"
MAX_STEPS = 1300

HOST = "127.0.0.1"
PORT = 8011

# Keep the same setting as our debug experiment.
EXEC_HORIZON = 50

OUTPUT_DIR = Path(
    "/home/cluster/smolvla_robomme_memory/"
    "outputs/robomme_eval_full_baseline_binary_gripper"
)

VIDEO_DIR = OUTPUT_DIR / "videos"
RESULTS_FILE = OUTPUT_DIR / "results.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utilities
# ============================================================

def load_results():
    if not RESULTS_FILE.exists():
        return {}

    try:
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Warning: failed to load existing results:", e)
        return {}


def save_results(results):
    tmp_file = RESULTS_FILE.with_suffix(".tmp")

    with open(tmp_file, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    tmp_file.replace(RESULTS_FILE)


def normalize_result(result):
    """
    Convert RoboMME result into a simple string:
        success
        fail
        ongoing
        timeout
        unknown
    """

    if result is None:
        return "unknown"

    # Handle strings directly.
    if isinstance(result, str):
        r = result.lower()

        if "success" in r:
            return "success"

        if "fail" in r:
            return "fail"

        if "ongoing" in r:
            return "ongoing"

        if "timeout" in r:
            return "timeout"

        return r

    return str(result)


# ============================================================
# Run one episode
# ============================================================

def run_episode(
    client,
    env_runner,
    episode_id,
):

    print()
    print("=" * 70)
    print(f"{TASK} | Episode {episode_id}")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    # Reset SmolVLA before EVERY episode.
    # This clears its internal action queue/state.
    # --------------------------------------------------------

    reset_result = client.reset()

    print("Policy reset:")
    print(reset_result)

    # --------------------------------------------------------
    # Create/reset RoboMME episode
    # --------------------------------------------------------

    env_runner.make_env(episode_id)

    print("Difficulty :", env_runner.difficulty)

    pre_traj = env_runner.get_init_obs()

    prompt = pre_traj["task_goal"]

    img = pre_traj["images"][-1]
    wrist_img = pre_traj["wrist_images"][-1]
    robot_state = pre_traj["states"][-1]

    print("Task goal:")
    print(prompt)

    print(
        "Initial obs:",
        np.asarray(img).shape,
        np.asarray(wrist_img).shape,
        np.asarray(robot_state).shape,
    )

    # --------------------------------------------------------
    # Recorder
    # --------------------------------------------------------

    recorder = RolloutRecorder(
        VIDEO_DIR,
        prompt,
        fps=30,
    )

    for i in range(len(pre_traj["images"])):
        recorder.record(
            image=pre_traj["images"][i].copy(),
            wrist_image=pre_traj["wrist_images"][i].copy(),
            state=pre_traj["states"][i].copy(),
            action=np.zeros(8, dtype=np.float32),
            subgoal=None,
        )

    # --------------------------------------------------------
    # Rollout
    # --------------------------------------------------------

    env_step = 0
    inference_id = 0

    final_result = "ongoing"

    try:

        while env_step < MAX_STEPS:

            # =================================================
            # Vanilla SmolVLA baseline:
            #
            # current image
            # current wrist image
            # current state
            # prompt
            #
            # NO history
            # NO memory
            # NO subgoal
            # =================================================

            element = {
                "observation/image": np.asarray(img),
                "observation/wrist_image": np.asarray(wrist_img),
                "observation/state": np.asarray(
                    robot_state,
                    dtype=np.float32,
                ),
                "prompt": prompt,
            }

            result = client.infer(element)

            if "actions" not in result:
                raise RuntimeError(
                    f"Server response has no 'actions': "
                    f"{result.keys()}"
                )

            actions = np.asarray(
                result["actions"],
                dtype=np.float32,
            )

            if actions.ndim != 2 or actions.shape[1] != 8:
                raise RuntimeError(
                    f"Unexpected action chunk shape: "
                    f"{actions.shape}. Expected [T, 8]."
                )

            if not np.all(np.isfinite(actions)):
                raise RuntimeError(
                    "Action chunk contains NaN or Inf."
                )

            actions = actions[:EXEC_HORIZON]

            inference_id += 1

            # =================================================
            # Execute
            # =================================================

            for action in actions:

                if env_step >= MAX_STEPS:
                    break

                action = np.asarray(
                    action,
                    dtype=np.float32,
                )

                # Diagnostic only: preserve the exact action sent to RoboMME.
                state_before = np.asarray(robot_state, dtype=np.float32).copy()
                action = np.asarray(action, dtype=np.float64).copy()
                action[7] = 1.0 if action[7] >= 0 else -1.0
                obs, stop_flag, outcome = env_runner.step(action)
                if obs[2] is not None and (env_step < 10 or env_step % 50 == 0):
                    state_after = np.asarray(obs[2], dtype=np.float32)
                    print("\n[ACTION TRACKING] step=", env_step)
                    print("target joints :", np.round(action[:7], 4))
                    print("before joints :", np.round(state_before[:7], 4))
                    print("after joints  :", np.round(state_after[:7], 4))
                    print("target - after:", np.round(action[:7] - state_after[:7], 4))
                    print("gripper cmd   :", round(float(action[7]), 4))
                    print("gripper before:", round(float(state_before[7]), 4))
                    print("gripper after :", round(float(state_after[7]), 4))


                current_result = normalize_result(outcome)

                if current_result != "ongoing":
                    final_result = current_result

                if env_step < 5 or env_step % 100 == 0:
                    print(
                        f"step={env_step:04d} "
                        f"status={current_result} "
                        f"gripper={action[-1]:+.3f}"
                    )

                env_step += 1

                # ---------------------------------------------
                # Invalid observation
                # ---------------------------------------------

                if obs[0] is None:

                    print(
                        "Environment returned invalid observation."
                    )

                    if final_result == "ongoing":
                        final_result = "fail"

                    return {
                        "episode": episode_id,
                        "prompt": str(prompt),
                        "result": final_result,
                        "env_steps": env_step,
                        "inferences": inference_id,
                    }

                img, wrist_img, robot_state = obs

                recorder.record(
                    image=img.copy(),
                    wrist_image=wrist_img.copy(),
                    state=robot_state.copy(),
                    action=action.copy(),
                    subgoal=None,
                )

                # ---------------------------------------------
                # Environment finished
                # ---------------------------------------------

                if stop_flag:

                    print(
                        f"Environment stop: {current_result}"
                    )

                    if current_result == "ongoing":
                        final_result = "unknown"
                    else:
                        final_result = current_result

                    return {
                        "episode": episode_id,
                        "prompt": str(prompt),
                        "result": final_result,
                        "env_steps": env_step,
                        "inferences": inference_id,
                    }

        # ----------------------------------------------------
        # Max steps reached
        # ----------------------------------------------------

        final_result = "timeout"

        return {
            "episode": episode_id,
            "prompt": str(prompt),
            "result": final_result,
            "env_steps": env_step,
            "inferences": inference_id,
        }

    finally:

        # ----------------------------------------------------
        # Save episode video
        # ----------------------------------------------------

        safe_prompt = (
            str(prompt)
            .replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )[:80]

        video_filename = (
            f"{TASK}_ep{episode_id:03d}_"
            f"{final_result}_{safe_prompt}.mp4"
        )

        try:

            recorder.save_video(video_filename)

            print(
                "Video:",
                VIDEO_DIR / video_filename,
            )

        except Exception as e:

            print(
                "Warning: failed to save video:",
                e,
            )

        # Important:
        # close current episode before next one.
        try:
            env_runner.close_env()
        except Exception as e:
            print(
                "Warning: failed to close environment:",
                e,
            )


# ============================================================
# Summary
# ============================================================

def print_summary(results, total_episodes):

    success = 0
    fail = 0
    timeout = 0
    unknown = 0

    for ep_result in results.values():

        status = ep_result.get(
            "result",
            "unknown",
        )

        if status == "success":
            success += 1

        elif status == "fail":
            fail += 1

        elif status == "timeout":
            timeout += 1

        else:
            unknown += 1

    completed = len(results)

    print()
    print("=" * 70)
    print("BinFill evaluation summary")
    print("=" * 70)

    print(
        f"Completed : {completed}/{total_episodes}"
    )

    print(
        f"Success   : {success}"
    )

    print(
        f"Fail      : {fail}"
    )

    print(
        f"Timeout   : {timeout}"
    )

    print(
        f"Unknown   : {unknown}"
    )

    if completed > 0:

        sr_completed = (
            success / completed * 100.0
        )

        print(
            f"SR completed episodes : "
            f"{sr_completed:.2f}%"
        )

    if completed == total_episodes:

        sr_full = (
            success / total_episodes * 100.0
        )

        print()
        print(
            f"FINAL BinFill SR: "
            f"{success}/{total_episodes} "
            f"= {sr_full:.2f}%"
        )

    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("SmolVLA -> RoboMME")
    print("Full BinFill TEST evaluation")
    print("=" * 70)

    print("Task         :", TASK)
    print("Server       :", f"ws://{HOST}:{PORT}")
    print("Max steps    :", MAX_STEPS)
    print("Exec horizon :", EXEC_HORIZON)

    print("Memory       : OFF")
    print("History      : OFF")
    print("Subgoal      : OFF")

    print("Results      :", RESULTS_FILE)
    print("Videos       :", VIDEO_DIR)

    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Connect once to SmolVLA server
    # --------------------------------------------------------

    print("Connecting to SmolVLA server...")

    client = MMEVLAWebsocketClientPolicy(
        host=HOST,
        port=PORT,
    )

    print("Connected.")

    print("Server metadata:")
    print(client.get_server_metadata())

    print()

    # --------------------------------------------------------
    # Create runner
    # --------------------------------------------------------

    env_runner = EnvRunner(
        TASK,
        str(VIDEO_DIR),
        max_steps=MAX_STEPS,
    )

    total_episodes = env_runner.num_episodes

    print(
        f"{TASK} test episodes:",
        total_episodes,
    )

    # --------------------------------------------------------
    # Existing results
    # --------------------------------------------------------

    results = load_results()

    if results:

        print(
            f"Found {len(results)} completed "
            "episodes in results.json."
        )

        print(
            "Those episodes will be skipped."
        )

    print_summary(
        results,
        total_episodes,
    )

    # --------------------------------------------------------
    # Run ALL test episodes
    # --------------------------------------------------------

    for episode_id in range(total_episodes):

        key = str(episode_id)

        # Resume support
        if key in results:

            print()
            print(
                f"[SKIP] Episode {episode_id} "
                "already completed."
            )

            continue

        try:

            episode_result = run_episode(
                client=client,
                env_runner=env_runner,
                episode_id=episode_id,
            )

        except KeyboardInterrupt:

            print()
            print(
                "Interrupted by user."
            )

            print(
                "Completed results have already "
                "been saved."
            )

            break

        except Exception as e:

            print()
            print(
                f"[ERROR] Episode {episode_id}:",
                repr(e),
            )

            episode_result = {
                "episode": episode_id,
                "prompt": "",
                "result": "error",
                "env_steps": 0,
                "inferences": 0,
                "error": repr(e),
            }

            # Try to close environment before next episode.
            try:
                env_runner.close_env()
            except Exception:
                pass

        # ----------------------------------------------------
        # Save immediately after every episode
        # ----------------------------------------------------

        results[key] = episode_result

        save_results(results)

        print()
        print(
            f"[SAVED] Episode {episode_id}: "
            f"{episode_result['result']}"
        )

        print_summary(
            results,
            total_episodes,
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print_summary(
        results,
        total_episodes,
    )

    print()
    print(
        "Results saved to:"
    )
    print(
        RESULTS_FILE
    )


if __name__ == "__main__":
    main()
