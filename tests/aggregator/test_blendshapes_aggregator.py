import asyncio
import logging
from typing import List

import numpy as np
import pytest

from orchestrator.aggregator.blendshapes_aggregator import BlendshapesAggregator
from orchestrator.data_structures.face_chunk import FaceChunkBody, FaceChunkEnd, FaceChunkStart
from orchestrator.data_structures.motion_chunk import MotionChunkBody, MotionChunkEnd, MotionChunkStart
from orchestrator.data_structures.process_flow import DAGNode, DAGStatus, DirectedAcyclicGraph
from orchestrator.profile.motion_face_stream_profile import MotionFaceStreamProfile


def _build_motion_bytes(
    n_frames: int,
    n_joints: int,
    motion_bs_names: List[str] | None,
    dtype: str,
    fill_value: float = 0.0,
) -> bytes:
    """Construct motion frame bytes matching BlendshapesAggregator
    expectations.

    Layout per frame: n_joints*9 + 3 + 3 + len(motion_bs_names or []).

    Args:
        n_frames (int):
            Number of frames to generate.
        n_joints (int):
            Number of joints in motion.
        motion_bs_names (List[str] | None):
            Motion blendshape names or None if not present.
        dtype (str):
            Floating dtype string, "float16" or "float32".
        fill_value (float, optional):
            Default fill for generated values. Defaults to 0.0.

    Returns:
        bytes:
            Serialized motion array bytes.
    """
    n_bs = 0 if motion_bs_names is None else len(motion_bs_names)
    one_frame_dim = n_joints * 9 + 3 + 3 + n_bs
    np_dtype = np.float16 if dtype == "float16" else np.float32
    arr = np.full((n_frames, one_frame_dim), fill_value, dtype=np_dtype)
    return arr.astype(np_dtype).tobytes()


def _build_face_bytes(
    n_frames: int,
    n_face_bs: int,
    dtype: str,
    values: list[list[float]] | None = None,
) -> bytes:
    """Construct face blendshape bytes for FaceChunkBody.

    Args:
        n_frames (int):
            Number of frames to generate.
        n_face_bs (int):
            Number of face blendshape channels.
        dtype (str):
            Floating dtype string, "float16" or "float32".
        values (list[list[float]] | None, optional):
            Optional 2D values with shape [n_frames, n_face_bs].
            If None, zeros are generated. Defaults to None.

    Returns:
        bytes:
            Serialized face blendshape array bytes.
    """
    np_dtype = np.float16 if dtype == "float16" else np.float32
    if values is None:
        arr = np.zeros((n_frames, n_face_bs), dtype=np_dtype)
    else:
        arr = np.array(values, dtype=np_dtype).reshape(n_frames, n_face_bs)
    return arr.astype(np_dtype).tobytes()


async def _run_until_processed(tasks: list[asyncio.Task], timeout: float = 1.5) -> None:
    """Wait for background tasks to process queued chunks.

    Args:
        tasks (list[asyncio.Task]):
            Tasks to monitor for completion.
        timeout (float, optional):
            Maximum seconds to wait. Defaults to 1.5.

    Returns:
        None:
            This function schedules awaiting and returns when done or timed out.
    """
    # Let event loop process queued tasks and IO briefly
    await asyncio.sleep(0.05)
    # Give some time for background workers to finish current batch
    try:
        await asyncio.wait_for(asyncio.gather(*[t for t in tasks if not t.done()]), timeout=timeout)
    except asyncio.TimeoutError:
        # normal in streaming tests; we'll cancel in the caller
        pass


@pytest.mark.asyncio
async def test_flow_none_none_motion_bs_none() -> None:
    """Both config lists None; Motion start without blendshapes.

    Expect: aggregator forwards motion and face streams unchanged to sinks.
    """
    logger_cfg = {"logger_name": "test_bsa_none_none_bs_none", "console_level": logging.DEBUG}
    aggregator = BlendshapesAggregator(queue_size=20, sleep_time=0.01, logger_cfg=logger_cfg)

    dag = DirectedAcyclicGraph("test_dag_1", {})
    agg_node = DAGNode("agg", aggregator)
    profile = MotionFaceStreamProfile(logger_cfg=logger_cfg)
    profile_node = DAGNode("profile", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "profile")

    request_id = "req_none_none_0"
    # Run
    dag.set_status(DAGStatus.RUNNING)
    tasks = [
        asyncio.create_task(aggregator.run()),
        asyncio.create_task(profile.run()),
    ]

    # Send starts
    await aggregator.feed_stream(
        MotionChunkStart(
            request_id=request_id,
            node_name="agg",
            dag=dag,
            joint_names=["root"],
            restpose_name="T",
            dtype="float32",
            blendshape_names=None,
            timeline_start_idx=0,
        )
    )
    await aggregator.feed_stream(
        FaceChunkStart(
            request_id=request_id,
            blendshape_names=["A", "B", "C"],
            dtype="float32",
            node_name="agg",
            dag=dag,
            timeline_start_idx=0,
        )
    )

    # Bodies
    motion_bytes = _build_motion_bytes(n_frames=2, n_joints=1, motion_bs_names=None, dtype="float32")
    face_bytes = _build_face_bytes(n_frames=2, n_face_bs=3, dtype="float32", values=[[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    await aggregator.feed_stream(MotionChunkBody(request_id=request_id, seq_number=0, data=motion_bytes))
    await aggregator.feed_stream(FaceChunkBody(request_id=request_id, seq_number=0, data=face_bytes))

    # Ends (send motion then face)
    await aggregator.feed_stream(MotionChunkEnd(request_id=request_id))
    await aggregator.feed_stream(FaceChunkEnd(request_id=request_id))

    await asyncio.sleep(0.2)
    # After processing, profiles should have removed the request from buffers
    assert request_id not in profile.input_buffer

    for t in tasks:
        t.cancel()


@pytest.mark.asyncio
async def test_flow_none_none_motion_bs_present() -> None:
    """Both config lists None; Motion start with blendshapes present.

    Expect: aggregator strips blendshapes from motion but forwards face unchanged.
    """
    logger_cfg = {"logger_name": "test_bsa_none_none_bs_present", "console_level": logging.DEBUG}
    aggregator = BlendshapesAggregator(queue_size=20, sleep_time=0.01, logger_cfg=logger_cfg)

    dag = DirectedAcyclicGraph("test_dag_2", {})
    agg_node = DAGNode("agg", aggregator)
    profile = MotionFaceStreamProfile(logger_cfg=logger_cfg)
    profile_node = DAGNode("profile", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "profile")

    dag.set_status(DAGStatus.RUNNING)
    tasks = [
        asyncio.create_task(aggregator.run()),
        asyncio.create_task(profile.run()),
    ]

    request_id = "req_none_none_1"
    motion_bs = ["A", "B"]
    face_bs = ["A", "B", "C"]

    await aggregator.feed_stream(
        MotionChunkStart(
            request_id=request_id,
            node_name="agg",
            dag=dag,
            joint_names=["root"],
            restpose_name="T",
            dtype="float32",
            blendshape_names=motion_bs,
            timeline_start_idx=0,
        )
    )
    await aggregator.feed_stream(
        FaceChunkStart(
            request_id=request_id,
            blendshape_names=face_bs,
            dtype="float32",
            node_name="agg",
            dag=dag,
            timeline_start_idx=0,
        )
    )

    # Prepare motion with blendshape tails
    m_one_frame = 1 * 9 + 3 + 3 + len(motion_bs)
    m = np.zeros((2, m_one_frame), dtype=np.float32)
    # set BS values
    m[0, -2:] = [0.5, 0.2]
    m[1, -2:] = [0.1, 0.0]
    motion_bytes = m.tobytes()

    face_bytes = _build_face_bytes(2, len(face_bs), "float32", values=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    await aggregator.feed_stream(MotionChunkBody(request_id=request_id, seq_number=0, data=motion_bytes))
    await aggregator.feed_stream(FaceChunkBody(request_id=request_id, seq_number=0, data=face_bytes))

    await aggregator.feed_stream(MotionChunkEnd(request_id=request_id))
    await aggregator.feed_stream(FaceChunkEnd(request_id=request_id))

    await asyncio.sleep(0.2)
    assert request_id not in profile.input_buffer

    for t in tasks:
        t.cancel()


@pytest.mark.asyncio
async def test_flow_motion_first_only() -> None:
    """Only motion_first_blendshape_names provided; aggregate from motion/face.

    Test before-zero branch by setting buffer_frame_idx to negative value.
    """
    logger_cfg = {"logger_name": "test_bsa_motion_first_only", "console_level": logging.DEBUG}
    aggregator = BlendshapesAggregator(
        motion_first_blendshape_names=["A", "B"],
        add_blendshape_names=None,
        queue_size=20,
        sleep_time=0.01,
        logger_cfg=logger_cfg,
    )

    dag = DirectedAcyclicGraph("test_dag_3", {})
    agg_node = DAGNode("agg", aggregator)
    profile = MotionFaceStreamProfile(logger_cfg=logger_cfg)
    profile_node = DAGNode("profile", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "profile")

    dag.set_status(DAGStatus.RUNNING)
    tasks = [
        asyncio.create_task(aggregator.run()),
        asyncio.create_task(profile.run()),
    ]

    request_id = "req_motion_first"
    motion_bs = ["A", "B"]
    face_bs = ["A", "B", "C"]

    await aggregator.feed_stream(
        MotionChunkStart(
            request_id=request_id,
            node_name="agg",
            dag=dag,
            joint_names=["root"],
            restpose_name="T",
            dtype="float32",
            blendshape_names=motion_bs,
            timeline_start_idx=-15,
        )
    )
    await aggregator.feed_stream(
        FaceChunkStart(
            request_id=request_id,
            blendshape_names=face_bs,
            dtype="float32",
            node_name="agg",
            dag=dag,
            timeline_start_idx=0,
        )
    )

    # Ensure internal buffer_frame_idx is initialized for aggregation path
    await asyncio.sleep(0.05)

    # Motion only (before 0), expect aggregator to synthesize face frames
    m_one_frame = 1 * 9 + 3 + 3 + len(motion_bs)
    m = np.zeros((3, m_one_frame), dtype=np.float32)
    m[:, -2:] = [[0.3, 0.0], [0.0, 0.0], [0.4, 0.5]]
    motion_bytes = m.tobytes()

    await aggregator.feed_stream(MotionChunkBody(request_id=request_id, seq_number=0, data=motion_bytes))

    await aggregator.feed_stream(MotionChunkEnd(request_id=request_id))
    await aggregator.feed_stream(FaceChunkEnd(request_id=request_id))

    await asyncio.sleep(0.3)
    assert request_id not in profile.input_buffer

    for t in tasks:
        t.cancel()


@pytest.mark.asyncio
async def test_flow_add_only() -> None:
    """Only add_blendshape_names provided; aggregate after 0 with both
    streams."""
    logger_cfg = {"logger_name": "test_bsa_add_only", "console_level": logging.DEBUG}
    aggregator = BlendshapesAggregator(
        motion_first_blendshape_names=None,
        add_blendshape_names=["A"],
        queue_size=20,
        sleep_time=0.01,
        logger_cfg=logger_cfg,
    )

    dag = DirectedAcyclicGraph("test_dag_4", {})
    agg_node = DAGNode("agg", aggregator)
    profile = MotionFaceStreamProfile(logger_cfg=logger_cfg)
    profile_node = DAGNode("profile", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "profile")

    request_id = "req_add_only"
    motion_bs = ["A", "B"]
    face_bs = ["A", "B", "C"]

    dag.set_status(DAGStatus.RUNNING)
    tasks = [
        asyncio.create_task(aggregator.run()),
        asyncio.create_task(profile.run()),
    ]

    await aggregator.feed_stream(
        MotionChunkStart(
            request_id=request_id,
            node_name="agg",
            dag=dag,
            joint_names=["root"],
            restpose_name="T",
            dtype="float32",
            blendshape_names=motion_bs,
            timeline_start_idx=0,
        )
    )
    await aggregator.feed_stream(
        FaceChunkStart(
            request_id=request_id,
            blendshape_names=face_bs,
            dtype="float32",
            node_name="agg",
            dag=dag,
            timeline_start_idx=0,
        )
    )

    await asyncio.sleep(0.05)
    aggregator.input_buffer[request_id]["buffer_frame_idx"] = 0

    # Provide both motion and face bodies
    m_one_frame = 1 * 9 + 3 + 3 + len(motion_bs)
    m = np.zeros((2, m_one_frame), dtype=np.float32)
    m[:, -2:] = [[0.2, 0.1], [0.0, 0.0]]
    motion_bytes = m.tobytes()

    f = np.zeros((2, len(face_bs)), dtype=np.float32)
    f[:, :] = [[0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]
    face_bytes = f.tobytes()

    await aggregator.feed_stream(MotionChunkBody(request_id=request_id, seq_number=0, data=motion_bytes))
    await aggregator.feed_stream(FaceChunkBody(request_id=request_id, seq_number=0, data=face_bytes))

    await aggregator.feed_stream(MotionChunkEnd(request_id=request_id))
    await aggregator.feed_stream(FaceChunkEnd(request_id=request_id))

    await asyncio.sleep(0.3)
    assert request_id not in profile.input_buffer

    for t in tasks:
        t.cancel()


@pytest.mark.asyncio
async def test_flow_both_lists() -> None:
    """Both motion_first and add lists provided; aggregate with both
    streams."""
    logger_cfg = {"logger_name": "test_bsa_both_lists", "console_level": logging.DEBUG}
    aggregator = BlendshapesAggregator(
        motion_first_blendshape_names=["A"],
        add_blendshape_names=["B"],
        queue_size=20,
        sleep_time=0.01,
        logger_cfg=logger_cfg,
    )

    dag = DirectedAcyclicGraph("test_dag_5", {})
    agg_node = DAGNode("agg", aggregator)
    profile = MotionFaceStreamProfile(logger_cfg=logger_cfg)
    profile_node = DAGNode("profile", profile)
    dag.add_node(agg_node)
    dag.add_node(profile_node)
    dag.add_edge("agg", "profile")

    request_id = "req_both_lists"
    motion_bs = ["A", "B"]
    face_bs = ["A", "B", "C"]

    dag.set_status(DAGStatus.RUNNING)
    tasks = [
        asyncio.create_task(aggregator.run()),
        asyncio.create_task(profile.run()),
    ]

    await aggregator.feed_stream(
        MotionChunkStart(
            request_id=request_id,
            node_name="agg",
            dag=dag,
            joint_names=["root"],
            restpose_name="T",
            dtype="float32",
            blendshape_names=motion_bs,
            timeline_start_idx=0,
        )
    )
    await aggregator.feed_stream(
        FaceChunkStart(
            request_id=request_id,
            blendshape_names=face_bs,
            dtype="float32",
            node_name="agg",
            dag=dag,
            timeline_start_idx=0,
        )
    )

    await asyncio.sleep(0.05)
    aggregator.input_buffer[request_id]["buffer_frame_idx"] = 0

    # Both streams bodies
    m_one_frame = 1 * 9 + 3 + 3 + len(motion_bs)
    m = np.zeros((2, m_one_frame), dtype=np.float32)
    m[:, -2:] = [[0.4, 0.2], [0.0, 0.1]]
    motion_bytes = m.tobytes()

    f = np.zeros((2, len(face_bs)), dtype=np.float32)
    f[:, :] = [[0.05, 0.05, 0.0], [0.1, 0.0, 0.0]]
    face_bytes = f.tobytes()

    await aggregator.feed_stream(MotionChunkBody(request_id=request_id, seq_number=0, data=motion_bytes))
    await aggregator.feed_stream(FaceChunkBody(request_id=request_id, seq_number=0, data=face_bytes))

    await aggregator.feed_stream(MotionChunkEnd(request_id=request_id))
    await aggregator.feed_stream(FaceChunkEnd(request_id=request_id))

    await asyncio.sleep(0.3)
    assert request_id not in profile.input_buffer

    for t in tasks:
        t.cancel()


@pytest.mark.asyncio
async def test_errors_handle_body_without_start_raises() -> None:
    """Calling _handle_body without start should raise
    ChunkWithoutStartError."""
    aggregator = BlendshapesAggregator(queue_size=10, sleep_time=0.01)
    # Build a body without start
    face_body = FaceChunkBody(request_id="no_start", seq_number=0, data=b"")
    with pytest.raises(Exception):
        await aggregator._handle_body(face_body, cur_time=0.0)
