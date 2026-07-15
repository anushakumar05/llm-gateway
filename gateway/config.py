import os

PROVIDERS = {
    "mock-a": {
        "type": "openai_compatible",
        "base_url": os.getenv("MOCK_A_URL", "http://localhost:9001"),
    },
    "mock-b": {
        "type": "openai_compatible",
        "base_url": os.getenv("MOCK_B_URL", "http://localhost:9002"),
    },
}

# model -> ordered list of providers to try. Phase 3 walks this chain.
MODEL_ROUTES = {
    "gpt-4o-mini": ["mock-a", "mock-b"],
    "gpt-4o": ["mock-a", "mock-b"],
}

DEFAULT_ROUTE = ["mock-a"]