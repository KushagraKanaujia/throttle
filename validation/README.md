# Historical schema-1 artifacts

The JSON files in this directory are sanitized local/fake-server artifacts from
Throttle 0.1. They are preserved as historical validation evidence and may use
superseded `recommendation` field names. They are smoke-only, are not current
production guidance, and are rejected by Throttle 0.2 saved-run comparison.

`fake_openai_server.py` is a local fixture, not a live-model benchmark. It is
maintained against the current strict schema-2 response/stream contract; the
historical JSON files were not regenerated or rewritten.
