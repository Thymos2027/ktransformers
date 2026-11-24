"""Short SSH outages are retried within the unchanged remote heartbeat deadline."""
HEARTBEAT_GRACE_SECONDS = 75
REMOTE_HEARTBEAT_TIMEOUT_SECONDS = 90


class TransportGrace:
    def __init__(self, now):
        self.last_success = now
        self.failures = 0

    def success(self, now):
        self.last_success = now
        self.failures = 0

    def failure(self, now):
        self.failures += 1
        return now - self.last_success >= HEARTBEAT_GRACE_SECONDS

    def timeout(self, now):
        return max(1, min(20, HEARTBEAT_GRACE_SECONDS - (now - self.last_success)))
