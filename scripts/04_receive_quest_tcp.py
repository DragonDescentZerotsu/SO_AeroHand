from hand_tracking_sdk import (
    HandFrame,
    HeadFrame,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)

client = HTSClient(
    HTSClientConfig(
        transport_mode=TransportMode.TCP_SERVER,
        host="0.0.0.0",
        port=8000,
        output=StreamOutput.FRAMES,
    )
)

print("Waiting for Quest TCP connection on port 8000...")

for frame in client.iter_events():
    if isinstance(frame, HeadFrame):
        print(
            f"[HEAD] seq={frame.sequence_id} "
            f"pos=({frame.head.x:.3f}, {frame.head.y:.3f}, {frame.head.z:.3f})"
        )
        continue

    if isinstance(frame, HandFrame):
        print(
            f"[HAND] side={frame.side.value} "
            f"seq={frame.sequence_id} "
            f"wrist=({frame.wrist.x:.3f}, {frame.wrist.y:.3f}, {frame.wrist.z:.3f}) "
            f"landmarks={len(frame.landmarks.points)}"
        )
