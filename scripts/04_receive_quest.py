from hand_tracking_sdk import HTSClient, HTSClientConfig, StreamOutput

client = HTSClient(
    HTSClientConfig(
        output=StreamOutput.BOTH,
    )
)

for event in client.iter_events():
    print(event)
